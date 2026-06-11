"""ArtiFixer3D WSL2/Docker adapter.

Mirrors ``omniroam_adapter``'s subprocess-orchestration pattern (path conversion,
pre-flight validation, streamed logs) but drives ArtiFixer3D as a sequence of
``docker run`` steps inside WSL — it does NOT import any ArtiFixer library on the
host. Every container contract here was validated in Phase 0/1
(``experiments/artifixer_eval/RESULT.md`` + ``PHASE1.md``).
"""

from __future__ import annotations

import subprocess
from pathlib import PureWindowsPath


# ---------------------------------------------------------------------------
# Path conversion (ported from omniroam_adapter.windows_to_wsl_path)
# ---------------------------------------------------------------------------

def windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to the corresponding WSL mount path.

    >>> windows_to_wsl_path(r"D:\\SPAG-4D\\output")
    '/mnt/d/SPAG-4D/output'
    """
    p = PureWindowsPath(win_path)
    parts = list(p.parts)
    if not parts:
        raise ValueError(f"Cannot convert empty path: {win_path!r}")

    drive = p.drive.rstrip(":\\").lower()
    if not drive:
        raise ValueError(f"Path has no drive letter: {win_path!r}")

    rest = "/".join(parts[1:])
    wsl_path = f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    while "//" in wsl_path:
        wsl_path = wsl_path.replace("//", "/")
    return wsl_path.rstrip("/") if wsl_path != "/" else "/"


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def _configs_dir(config) -> str:
    """WSL path to the 3DGRUT configs dir where the hydra wrappers live."""
    return f"{config.artifixer_repo}/thirdparty/3DGRUT-ArtiFixer/configs"


def validate_artifixer_environment(config) -> None:
    """Pre-flight checks confirming the ArtiFixer3D WSL/Docker stack is ready.

    Probes (each via ``wsl -d <distro> bash -c``):
        (a) GPU visible inside the container (``docker run --gpus all ... nvidia-smi``)
        (b) the 14B checkpoint file exists
        (c) the local Wan2.1 mirror is populated (text_encoder dir + vae weights)
        (d) both hydra wrapper yamls are copied into the repo configs dir

    Raises
    ------
    RuntimeError
        With a fix hint (pointing at setup_wan_mirror.sh / INSTALL.md) on the
        first failed probe.
    """
    distro = config.wsl_distro

    # (a) GPU-in-container gate.
    inner = f"docker run --rm --gpus all {config.docker_image} nvidia-smi -L"
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", inner],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ArtiFixer3D: GPU not visible inside the container "
            f"(image '{config.docker_image}', distro '{distro}'). "
            "Confirm WSL2 + Docker + NVIDIA Container Toolkit per INSTALL.md. "
            f"stderr: {result.stderr.strip()}"
        )

    # (b) checkpoint present.
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", f"test -f {config.checkpoint}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ArtiFixer3D: 14B checkpoint not found at '{config.checkpoint}'. "
            "Place artifixer-14b.pt there (see INSTALL.md)."
        )

    # (c) local Wan2.1 mirror populated.
    mirror_probe = (
        f"test -d {config.wan_mirror}/text_encoder && "
        f"test -f {config.wan_mirror}/vae/diffusion_pytorch_model.safetensors"
    )
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", mirror_probe],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ArtiFixer3D: local Wan2.1 mirror incomplete at '{config.wan_mirror}'. "
            "Run `bash spag4d/refine/artifixer3d_resources/setup_wan_mirror.sh` once "
            "inside WSL (see INSTALL.md)."
        )

    # (d) hydra wrapper configs copied into the repo.
    cfg_dir = _configs_dir(config)
    cfg_probe = (
        f"test -f {cfg_dir}/_artifixer_run.yaml && "
        f"test -f {cfg_dir}/_artifixer_distill.yaml"
    )
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", cfg_probe],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ArtiFixer3D: hydra wrapper configs missing in '{cfg_dir}'. "
            "Copy _artifixer_run.yaml + _artifixer_distill.yaml from "
            "spag4d/refine/artifixer3d_resources/ there (see INSTALL.md)."
        )
