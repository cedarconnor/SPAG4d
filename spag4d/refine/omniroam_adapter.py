"""OmniRoam WSL2 adapter.

Wraps OmniRoam (runs inside a WSL2 Linux environment) so SPAG-4D (Windows)
can invoke it via subprocess.  Handles path conversion, pre-flight validation,
streaming progress, and output video decoding.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Callable, List, Optional

# cv2 is an optional dependency — imported lazily in extract_video_frames.


# ---------------------------------------------------------------------------
# Path conversion
# ---------------------------------------------------------------------------

def windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to the corresponding WSL mount path.

    Examples
    --------
    >>> windows_to_wsl_path(r"D:\\SPAG-4D\\output")
    '/mnt/d/SPAG-4D/output'
    >>> windows_to_wsl_path("C:/Users/Cedar/file.jpg")
    '/mnt/c/Users/Cedar/file.jpg'
    """
    # Use PureWindowsPath so this also works when called from Linux (tests).
    p = PureWindowsPath(win_path)

    # Strip trailing separators by reconstructing from parts.
    parts = list(p.parts)
    if not parts:
        raise ValueError(f"Cannot convert empty path: {win_path!r}")

    # parts[0] is the drive root, e.g. "D:\\" → drive letter = "D"
    drive = p.drive.rstrip(":\\").lower()
    if not drive:
        raise ValueError(f"Path has no drive letter: {win_path!r}")

    # Remaining parts after the drive root (p.parts[0] is "D:\\")
    rest = "/".join(parts[1:])

    wsl_path = f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"

    # Normalise double slashes that could arise from trailing separators.
    while "//" in wsl_path:
        wsl_path = wsl_path.replace("//", "/")

    return wsl_path.rstrip("/") if wsl_path != "/" else "/"


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def validate_wsl_environment(config) -> None:  # config: OmniRoamConfig
    """Run pre-flight checks to confirm WSL2 + OmniRoam are ready.

    Raises
    ------
    RuntimeError
        With a descriptive message explaining what is missing.
    """
    distro = config.wsl_distro

    # 1. Check the requested WSL distro exists and responds.
    result = subprocess.run(
        ["wsl", "-d", distro, "echo", "ok"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"WSL distro '{distro}' not found or not running. "
            f"stderr: {result.stderr.strip()}"
        )

    # 2. Check conda env has torch (proxy for OmniRoam dependencies).
    result = subprocess.run(
        [
            "wsl", "-d", distro, "bash", "-c",
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate omniroam && python -c 'import torch'",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "conda environment 'omniroam' is missing or 'torch' is not installed. "
            f"stderr: {result.stderr.strip()}"
        )

    # 3. Check infer_omniroam.py exists at the configured install dir.
    install_dir = config.install_dir
    result = subprocess.run(
        [
            "wsl", "-d", distro, "bash", "-c",
            f"test -f {install_dir}/infer_omniroam.py",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OmniRoam not found at '{install_dir}/infer_omniroam.py' "
            f"inside WSL distro '{distro}'."
        )


# ---------------------------------------------------------------------------
# Main inference runner
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(r"(\d+)/(\d+)")


def run_omniroam_wsl(
    image_path: str,
    output_dir: str,
    preset: str,
    config,  # OmniRoamConfig
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Run OmniRoam inside WSL2 and stream stdout in real time.

    Parameters
    ----------
    image_path:
        Windows path to the source panorama image.
    output_dir:
        Windows path to the directory where output should be written.
    preset:
        Trajectory preset name passed to OmniRoam (e.g. "forward").
    config:
        An ``OmniRoamConfig`` instance.
    progress_callback:
        Optional callable ``(current: int, total: int) -> None`` fired
        whenever a ``"N/M"`` pattern is found in OmniRoam's output.

    Returns
    -------
    str
        The ``output_dir`` path (Windows) on success.

    Raises
    ------
    RuntimeError
        If OmniRoam exits with a non-zero status, with the last 20 lines of
        combined stdout/stderr included for debugging.
    """
    # Stage the source image into a temp input subdirectory.
    # OmniRoam expects --local_images_dir to be a *directory*.
    src = Path(image_path)
    input_staging_dir = Path(output_dir) / "_omniroam_input"
    input_staging_dir.mkdir(parents=True, exist_ok=True)
    staged_src = input_staging_dir / src.name
    shutil.copy2(src, staged_src)

    # Convert Windows paths to WSL mount paths.
    wsl_input_dir = windows_to_wsl_path(str(input_staging_dir))
    wsl_output_dir = windows_to_wsl_path(str(output_dir))

    distro = config.wsl_distro
    install_dir = config.install_dir

    # Build the inner shell command that runs inside WSL bash.
    # Source conda init so the non-interactive shell can find conda.
    inner_cmd = (
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate omniroam && "
        f"cd {install_dir} && "
        f"python infer_omniroam.py "
        f"--local_images_dir {wsl_input_dir} "
        f"--output_dir {wsl_output_dir} "
        f"--ckpt_path {config.ckpt_path} "
        f"--use_cam_traj --traj_mode fixed "
        f"--traj_preset {preset} "
        f"--re_scale_pose fixed:1.0 "
        f"--enable_speed_control --speed_fixed {config.speed} "
        f"--height {config.height} "
        f"--width {config.width} "
        f"--num_frames {config.num_frames} "
        f"--cfg_scale {config.cfg_scale} "
        f"--num_inference_steps {config.inference_steps} "
        f"--device cuda:0"
    )

    cmd = ["wsl", "-d", distro, "bash", "-c", inner_cmd]

    log_lines: List[str] = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    for line in proc.stdout:
        log_lines.append(line)
        if len(log_lines) > 20:
            log_lines.pop(0)

        if progress_callback is not None:
            match = _PROGRESS_RE.search(line)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                progress_callback(current, total)

    exit_code = proc.wait()

    if exit_code != 0:
        tail = "".join(log_lines)
        raise RuntimeError(
            f"OmniRoam failed with exit code {exit_code}.\n"
            f"Last output:\n{tail}"
        )

    return output_dir


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: str) -> List:
    """Read an mp4 video and return its frames as float32 RGB arrays.

    Parameters
    ----------
    video_path:
        Path to the mp4 file.

    Returns
    -------
    list of np.ndarray
        Each element has shape ``(H, W, 3)``, dtype ``float32``,
        values in ``[0.0, 1.0]``.
    """
    import cv2  # noqa: PLC0415  (lazy import — cv2 is optional)
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
    finally:
        cap.release()

    return frames
