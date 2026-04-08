"""Native Windows SeedVR2 adapter — subprocess wrapper for inference_cli.py.

Replaces the WSL2-based adapter in spag4d/refine/seedvr2_adapter.py.
Runs natively on Windows using sys.executable as the Python interpreter.
Supports both image mode (face upscaling) and video mode (OmniRoam output).
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(r"(\d+)/(\d+)")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SeedVR2Config:
    """Configuration for the native Windows SeedVR2 adapter."""

    install_dir: str = str(_PROJECT_ROOT / "third_party" / "seedvr2_videoupscaler")
    model: str = "seedvr2_ema_3b_fp16.safetensors"
    target_resolution: int = 1024
    batch_size: int = 1
    color_correction: str = "lab"
    block_swap: int = 0
    seed: int = 42
    attention_mode: str = "sdpa"


def validate_seedvr2_environment(config: SeedVR2Config) -> None:
    """Check that inference_cli.py exists in the install directory.

    Raises:
        FileNotFoundError: if inference_cli.py is not found at config.install_dir.
    """
    cli_path = Path(config.install_dir) / "inference_cli.py"
    if not cli_path.is_file():
        raise FileNotFoundError(
            f"SeedVR2 inference_cli.py not found at '{cli_path}'. "
            f"Install with: git clone "
            f"https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "
            f"{config.install_dir}"
        )
    logger.info("SeedVR2 environment validated OK: %s", cli_path)


def build_seedvr2_args(
    input_path: str,
    output_path: str,
    config: SeedVR2Config,
    mode: str = "image",
) -> list:
    """Build the CLI argument list for inference_cli.py.

    Args:
        input_path: Path to input image or video.
        output_path: Path for output.
        config: SeedVR2Config instance.
        mode: "image" or "video".

    Returns:
        List of string arguments (does NOT include the python / script prefix).
    """
    args = [
        input_path,
        "--output", output_path,
        "--dit_model", config.model,
        "--resolution", str(config.target_resolution),
        "--color_correction", config.color_correction,
        "--attention_mode", config.attention_mode,
        "--batch_size", str(config.batch_size),
        "--seed", str(config.seed),
    ]

    if mode == "image":
        args += ["--output_format", "png"]
    elif mode == "video":
        args += ["--video_backend", "opencv"]

    if config.block_swap > 0:
        args += [
            "--blocks_to_swap", str(config.block_swap),
            "--dit_offload_device", "cpu",
        ]

    return args


def upscale_images(
    faces: Dict[str, np.ndarray],
    config: SeedVR2Config,
    temp_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Upscale a dict of face images using SeedVR2 image mode.

    Args:
        faces: Dict mapping face name to uint8 HWC numpy array.
        config: SeedVR2Config instance.
        temp_dir: Directory for intermediate PNG files.
        progress_callback: Optional (current, total) progress callback.

    Returns:
        Tuple of (upscaled_dict, new_width, new_height).
        upscaled_dict maps face name to upscaled uint8 HWC numpy array.

    Raises:
        FileNotFoundError: if inference_cli.py is missing.
        RuntimeError: if the subprocess fails.
    """
    from PIL import Image  # optional dep; only needed at call time

    validate_seedvr2_environment(config)

    tmp = Path(temp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    cli_path = str(Path(config.install_dir) / "inference_cli.py")

    upscaled: Dict[str, np.ndarray] = {}
    new_w = new_h = 0

    for idx, (name, img_array) in enumerate(faces.items()):
        in_path = str(tmp / f"{name}_in.png")
        out_path = str(tmp / f"{name}_out.png")

        Image.fromarray(img_array).save(in_path)

        cli_args = build_seedvr2_args(in_path, out_path, config, mode="image")
        cmd = [sys.executable, cli_path] + cli_args

        def _cb(cur: int, tot: int) -> None:
            if progress_callback is not None:
                # Map local progress into overall progress across all faces
                overall_cur = idx * tot + cur
                overall_tot = len(faces) * tot
                progress_callback(overall_cur, overall_tot)

        _run_seedvr2_subprocess(cmd, _cb)

        out_img = np.array(Image.open(out_path).convert("RGB"))
        upscaled[name] = out_img
        new_h, new_w = out_img.shape[:2]

    return upscaled, new_w, new_h


def upscale_video(
    video_path: str,
    output_path: str,
    config: SeedVR2Config,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Upscale a video using SeedVR2 video mode.

    Args:
        video_path: Path to input video.
        output_path: Path for the upscaled output video.
        config: SeedVR2Config instance.
        progress_callback: Optional (current_batch, total_batches) callback.

    Returns:
        output_path on success.

    Raises:
        FileNotFoundError: if inference_cli.py is missing.
        RuntimeError: if the subprocess fails.
    """
    validate_seedvr2_environment(config)

    cli_path = str(Path(config.install_dir) / "inference_cli.py")
    cli_args = build_seedvr2_args(video_path, output_path, config, mode="video")
    cmd = [sys.executable, cli_path] + cli_args

    logger.info(
        "Running SeedVR2 video upscale: resolution=%d, model=%s",
        config.target_resolution,
        config.model,
    )

    _run_seedvr2_subprocess(cmd, progress_callback)

    logger.info("SeedVR2 video upscale complete -> %s", output_path)
    return output_path


def _run_seedvr2_subprocess(
    cmd: list,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Shared subprocess runner with streaming output and progress tracking.

    Args:
        cmd: Full command list including python executable and script path.
        progress_callback: Optional (current, total) callback invoked on each
            progress line that matches ``\\d+/\\d+``.

    Raises:
        RuntimeError: if the process exits with a non-zero return code.
    """
    logger.debug("SeedVR2 cmd: %s", " ".join(str(c) for c in cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    tail_lines: list = []

    for line in proc.stdout:
        line_stripped = line.rstrip()
        logger.debug("[seedvr2] %s", line_stripped)

        tail_lines.append(line)
        if len(tail_lines) > 20:
            tail_lines.pop(0)

        if progress_callback is not None:
            match = _PROGRESS_RE.search(line)
            if match:
                progress_callback(int(match.group(1)), int(match.group(2)))

    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"SeedVR2 failed with exit code {returncode}.\n"
            f"Last output:\n{''.join(tail_lines)}"
        )
