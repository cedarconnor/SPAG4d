"""Subprocess adapter around UniSHARP's scripts/infer_unisharp.py."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def run_unisharp_inference(
    image_path: str,
    out_dir: str,
    repo_dir: str,
    checkpoint_path: str,
    python_exe: Optional[str] = None,
    camera: str = "panorama",
    device: str = "cuda:0",
    save_ply: bool = True,
    max_images: int = 1,
    extra_args: Optional[list] = None,
    timeout_s: Optional[float] = None,
) -> dict:
    """Invoke UniSHARP's infer_unisharp.py as a subprocess.

    Returns dict with returncode, stdout, stderr, out_dir, ply_path,
    metadata_path. Raises FileNotFoundError if the script or output PLY is
    missing, and RuntimeError on a non-zero exit.
    """
    repo = Path(repo_dir)
    script = repo / "scripts" / "infer_unisharp.py"
    if not script.exists():
        raise FileNotFoundError(f"UniSHARP inference script not found: {script}")

    python = python_exe or sys.executable

    cmd = [
        python, str(script),
        "--checkpoint", str(checkpoint_path),
        "--image", str(image_path),
        "--out-dir", str(out_dir),
        "--camera", camera,
        "--device", device,
        "--max-images", str(max_images),
    ]
    if save_ply:
        cmd.append("--save-ply")  # REQUIRED or no PLY is written
    if extra_args:
        cmd.extend(extra_args)

    LOGGER.info("Running UniSHARP: %s", " ".join(cmd))

    # cwd MUST be the repo root: the script does sys.path.insert(REPO_ROOT),
    # writes checkpoints/torchhub there, and expects ./UniK3D as a sibling.
    proc = subprocess.run(
        cmd, cwd=str(repo),
        capture_output=True, text=True, timeout=timeout_s,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "UniSHARP inference failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )

    out = Path(out_dir)
    ply_candidates = sorted(out.glob("**/gaussians.ply"))
    meta_candidates = sorted(out.glob("**/metadata.json"))
    if not ply_candidates:
        raise FileNotFoundError(
            f"UniSHARP completed but no gaussians.ply under {out}. "
            "Confirm --save-ply was passed and inference produced a sample dir."
        )
    if len(ply_candidates) > 1:
        LOGGER.warning("Multiple PLYs found; using first: %s", ply_candidates[0])

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_dir": str(out),
        "ply_path": str(ply_candidates[0]),
        "metadata_path": str(meta_candidates[0]) if meta_candidates else None,
    }
