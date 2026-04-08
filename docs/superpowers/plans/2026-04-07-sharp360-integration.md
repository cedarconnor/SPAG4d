# SHARP 360 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Apple's SHARP as a third splat generator with native SeedVR2 upscaling, replacing the WSL2 SeedVR2 adapter.

**Architecture:** A new `sharp360.py` orchestrates per-face SHARP prediction with DA360 alignment and merge. A new `seedvr2.py` provides native Windows image+video upscaling, replacing the WSL2 adapter. `core.py` dispatches between DA360/DAP (existing SPAG path) and SHARP 360 (new path). All generators produce the same PLY format, compatible with existing refinement backends.

**Tech Stack:** PyTorch, Apple ml-sharp (vendored), SeedVR2 inference_cli.py (native subprocess), DA360 (existing), FastAPI, GaussianSplats3D viewer.

---

### Task 1: Native SeedVR2 Adapter

**Files:**
- Create: `spag4d/seedvr2.py`
- Create: `tests/test_seedvr2_native.py`

- [ ] **Step 1: Write failing tests for native SeedVR2 adapter**

```python
# tests/test_seedvr2_native.py
"""Tests for native Windows SeedVR2 adapter."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_build_image_args():
    """Build CLI args for image upscale mode."""
    from spag4d.seedvr2 import build_seedvr2_args, SeedVR2Config

    cfg = SeedVR2Config()
    args = build_seedvr2_args(
        input_path="/tmp/input",
        output_path="/tmp/output",
        config=cfg,
        mode="image",
    )
    assert "/tmp/input" in args
    assert "--output" in args
    assert "--dit_model" in args
    assert "--color_correction" in args
    assert "--output_format" in args
    # Image mode should have png output format
    idx = args.index("--output_format")
    assert args[idx + 1] == "png"


def test_build_video_args():
    """Build CLI args for video upscale mode."""
    from spag4d.seedvr2 import build_seedvr2_args, SeedVR2Config

    cfg = SeedVR2Config()
    args = build_seedvr2_args(
        input_path="/tmp/input.mp4",
        output_path="/tmp/output.mp4",
        config=cfg,
        mode="video",
    )
    assert "--video_backend" in args
    idx = args.index("--video_backend")
    assert args[idx + 1] == "opencv"


def test_build_args_with_block_swap():
    """When block_swap > 0, dit_offload_device should be cpu."""
    from spag4d.seedvr2 import build_seedvr2_args, SeedVR2Config

    cfg = SeedVR2Config(block_swap=32)
    args = build_seedvr2_args("/tmp/in", "/tmp/out", cfg, "image")
    assert "--dit_offload_device" in args
    idx = args.index("--dit_offload_device")
    assert args[idx + 1] == "cpu"
    assert "--blocks_to_swap" in args
    idx2 = args.index("--blocks_to_swap")
    assert args[idx2 + 1] == "32"


def test_build_args_no_block_swap():
    """When block_swap == 0, no offload flags."""
    from spag4d.seedvr2 import build_seedvr2_args, SeedVR2Config

    cfg = SeedVR2Config(block_swap=0)
    args = build_seedvr2_args("/tmp/in", "/tmp/out", cfg, "image")
    assert "--blocks_to_swap" not in args
    assert "--dit_offload_device" not in args


def test_config_defaults():
    """SeedVR2Config has sensible defaults."""
    from spag4d.seedvr2 import SeedVR2Config

    cfg = SeedVR2Config()
    assert cfg.model == "seedvr2_ema_3b_fp16.safetensors"
    assert cfg.color_correction == "lab"
    assert cfg.block_swap == 0
    assert cfg.target_resolution == 1024


def test_validate_environment_missing(tmp_path):
    """Validation fails when inference_cli.py is missing."""
    from spag4d.seedvr2 import validate_seedvr2_environment, SeedVR2Config

    cfg = SeedVR2Config(install_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError):
        validate_seedvr2_environment(cfg)


def test_validate_environment_present(tmp_path):
    """Validation passes when inference_cli.py exists."""
    from spag4d.seedvr2 import validate_seedvr2_environment, SeedVR2Config

    install_dir = tmp_path / "seedvr2"
    install_dir.mkdir()
    (install_dir / "inference_cli.py").write_text("# stub")
    cfg = SeedVR2Config(install_dir=str(install_dir))
    validate_seedvr2_environment(cfg)  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_seedvr2_native.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spag4d.seedvr2'`

- [ ] **Step 3: Implement native SeedVR2 adapter**

```python
# spag4d/seedvr2.py
"""Native Windows SeedVR2 adapter — image and video upscaling.

Replaces the WSL2-based seedvr2_adapter. Calls inference_cli.py directly
via subprocess using the current Python interpreter.
"""

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(r"(\d+)/(\d+)")

# Default install location (relative to project root)
_DEFAULT_INSTALL_DIR = str(
    Path(__file__).resolve().parents[1] / "third_party" / "seedvr2_videoupscaler"
)


@dataclass
class SeedVR2Config:
    """Configuration for native SeedVR2 upscaling."""

    install_dir: str = _DEFAULT_INSTALL_DIR
    model: str = "seedvr2_ema_3b_fp16.safetensors"
    target_resolution: int = 1024
    batch_size: int = 1
    color_correction: str = "lab"
    block_swap: int = 0
    seed: int = 42
    attention_mode: str = "sdpa"


def validate_seedvr2_environment(config: SeedVR2Config) -> None:
    """Check that inference_cli.py exists at the configured install dir.

    Raises FileNotFoundError if not found.
    """
    cli_path = Path(config.install_dir) / "inference_cli.py"
    if not cli_path.exists():
        raise FileNotFoundError(
            f"SeedVR2 not found at '{cli_path}'. "
            f"Install SeedVR2 into '{config.install_dir}' or run: "
            f"python -m spag4d download-models --model seedvr2"
        )
    logger.info("SeedVR2 environment validated OK at %s", config.install_dir)


def build_seedvr2_args(
    input_path: str,
    output_path: str,
    config: SeedVR2Config,
    mode: str = "image",
) -> list:
    """Build CLI argument list for inference_cli.py.

    Args:
        input_path: Path to input (directory for images, file for video).
        output_path: Path for output.
        config: SeedVR2Config with model/resolution settings.
        mode: "image" or "video".

    Returns:
        List of CLI argument strings.
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
    temp_dir: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple:
    """Upscale a dict of face images using SeedVR2.

    Args:
        faces: {name: (H, W, 3) uint8 array} face images.
        config: SeedVR2Config.
        temp_dir: Working directory for input/output PNGs.
        progress_callback: Optional (current, total) callback.

    Returns:
        (upscaled_faces, new_width, new_height) where upscaled_faces
        is {name: (H', W', 3) uint8 array}.
    """
    validate_seedvr2_environment(config)

    input_dir = temp_dir / "seedvr2_input"
    output_dir = temp_dir / "seedvr2_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write input faces
    for name, img_array in faces.items():
        Image.fromarray(img_array).save(input_dir / f"{name}.png")

    cli_path = str(Path(config.install_dir) / "inference_cli.py")
    cli_args = build_seedvr2_args(
        str(input_dir), str(output_dir), config, mode="image",
    )

    cmd = [sys.executable, cli_path] + cli_args
    logger.info("Running SeedVR2 image upscale: %d faces, target=%dp",
                len(faces), config.target_resolution)

    _run_seedvr2_subprocess(cmd, progress_callback)

    # Read back upscaled faces
    upscaled = {}
    for name in faces:
        out_path = output_dir / f"{name}.png"
        if not out_path.exists():
            raise FileNotFoundError(
                f"SeedVR2 did not produce expected output: {out_path}"
            )
        with Image.open(out_path) as img:
            upscaled[name] = np.asarray(img.convert("RGB")).copy()

    first = upscaled[next(iter(upscaled))]
    new_h, new_w = first.shape[:2]
    logger.info("SeedVR2 image upscale complete. New face size: %dx%d", new_w, new_h)
    return upscaled, new_w, new_h


def upscale_video(
    video_path: str,
    output_path: str,
    config: SeedVR2Config,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Upscale a video file using SeedVR2.

    Args:
        video_path: Path to input video.
        output_path: Path for upscaled output video.
        config: SeedVR2Config.
        progress_callback: Optional (current, total) callback.

    Returns:
        output_path on success.
    """
    validate_seedvr2_environment(config)

    cli_path = str(Path(config.install_dir) / "inference_cli.py")
    cli_args = build_seedvr2_args(video_path, output_path, config, mode="video")

    cmd = [sys.executable, cli_path] + cli_args
    logger.info("Running SeedVR2 video upscale: %s -> %dp",
                video_path, config.target_resolution)

    _run_seedvr2_subprocess(cmd, progress_callback)

    logger.info("SeedVR2 video upscale complete: %s", output_path)
    return output_path


def _run_seedvr2_subprocess(
    cmd: list,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Run SeedVR2 subprocess with streaming output and progress tracking."""
    log_lines = []
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
                progress_callback(int(match.group(1)), int(match.group(2)))

    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"SeedVR2 failed with exit code {returncode}.\n"
            f"Last output:\n{''.join(log_lines)}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_seedvr2_native.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/seedvr2.py tests/test_seedvr2_native.py
git commit -m "feat: add native Windows SeedVR2 adapter (image + video)"
```

---

### Task 2: Rewire OmniRoam Pipeline to Native SeedVR2

**Files:**
- Modify: `spag4d/refine/pipeline_v2.py:30,372-409`
- Modify: `spag4d/refine/omniroam_config.py:42-49`
- Remove: `spag4d/refine/seedvr2_adapter.py`
- Remove: `tests/test_seedvr2_adapter.py`

- [ ] **Step 1: Update omniroam_config.py SeedVR2 fields for native path**

Replace the WSL2 SeedVR2 fields in `spag4d/refine/omniroam_config.py:42-49`:

```python
    # ── Upscale (optional) ──
    upscale_backend: str = "none"  # "none" | "seedvr2"
    seedvr2_model: str = "seedvr2_ema_3b_fp16.safetensors"
    seedvr2_target_resolution: int = 1024  # short-side pixels (2x from 480)
    seedvr2_color_correction: str = "lab"
    seedvr2_block_swap: int = 0  # 0 = no offload (native has enough RAM typically)
```

Remove these WSL2-specific fields:
- `seedvr2_install_dir` (now comes from SeedVR2Config default)
- `seedvr2_batch_size` (now in SeedVR2Config)

- [ ] **Step 2: Rewire pipeline_v2.py Stage 3 to native SeedVR2**

In `spag4d/refine/pipeline_v2.py`, replace the import at line 30:

```python
# Old:
from .seedvr2_adapter import validate_seedvr2_environment, run_seedvr2_upscale

# New:
from ..seedvr2 import upscale_video as seedvr2_upscale_video, SeedVR2Config, validate_seedvr2_environment
```

Replace Stage 3 body (lines 372-409) with:

```python
    if config.upscale_backend == "seedvr2" and omniroam_frames_by_traj:
        logger.info(f"[Stage 3] Upscaling with SeedVR2 "
                     f"(resolution={config.seedvr2_target_resolution})")
        try:
            seedvr2_cfg = SeedVR2Config(
                model=config.seedvr2_model,
                target_resolution=config.seedvr2_target_resolution,
                color_correction=config.seedvr2_color_correction,
                block_swap=config.seedvr2_block_swap,
            )
            validate_seedvr2_environment(seedvr2_cfg)

            for preset in list(omniroam_frames_by_traj.keys()):
                report(f"upscale_{preset}", 35)

                traj_dir = work_dir / preset
                videos = list(traj_dir.rglob("generated.mp4"))
                if not videos:
                    logger.warning(f"[Stage 3] No video found for preset={preset}, skipping upscale")
                    continue

                src_video = str(videos[0])
                upscaled_video = str(videos[0].parent / "generated_upscaled.mp4")
                logger.info(f"[Stage 3] Upscaling {src_video}")

                seedvr2_upscale_video(
                    video_path=src_video,
                    output_path=upscaled_video,
                    config=seedvr2_cfg,
                )

                upscaled_frames = extract_video_frames(upscaled_video)
                if upscaled_frames:
                    logger.info(f"[Stage 3] Upscaled {preset}: {len(upscaled_frames)} frames "
                                f"at {upscaled_frames[0].shape[1]}x{upscaled_frames[0].shape[0]}")
                    omniroam_frames_by_traj[preset] = upscaled_frames
                else:
                    logger.warning(f"[Stage 3] Failed to extract upscaled frames for {preset}")
        except Exception as e:
            logger.error(f"[Stage 3] SeedVR2 upscale FAILED: {e}", exc_info=True)
            raise
```

- [ ] **Step 3: Delete WSL2 SeedVR2 adapter and its tests**

```bash
git rm spag4d/refine/seedvr2_adapter.py
git rm tests/test_seedvr2_adapter.py
```

- [ ] **Step 4: Run existing tests to verify nothing breaks**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v --ignore=tests/test_seedvr2_adapter.py`
Expected: All tests PASS (the import in pipeline_v2.py may need adjustment if tests import it)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: replace WSL2 SeedVR2 adapter with native Windows version"
```

---

### Task 3: Vendor ml-sharp

**Files:**
- Create: `spag4d/sharp_arch/ml-sharp/` (copy from SHARP_360_to_Splat repo)

- [ ] **Step 1: Copy ml-sharp source into vendor directory**

```bash
mkdir -p spag4d/sharp_arch
cp -r /tmp/SHARP_360_review/ml-sharp spag4d/sharp_arch/ml-sharp
```

If the clone is unavailable, clone fresh:

```bash
cd /tmp && git clone --depth 1 https://github.com/Enndee/SHARP_360_to_Splat.git SHARP_360_review
cp -r /tmp/SHARP_360_review/ml-sharp spag4d/sharp_arch/ml-sharp
```

- [ ] **Step 2: Verify SHARP imports work**

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'spag4d/sharp_arch/ml-sharp/src')
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D, save_ply, apply_transform
from sharp.utils.color_space import linearRGB2sRGB
print('All SHARP imports OK')
"
```

Expected: `All SHARP imports OK` (may need `pip install plyfile` if not already installed)

- [ ] **Step 3: Install any missing dependencies**

```bash
.venv/Scripts/pip.exe install plyfile
```

- [ ] **Step 4: Commit vendored ml-sharp**

```bash
git add spag4d/sharp_arch/ml-sharp/
git commit -m "vendor: add Apple ml-sharp for SHARP 360 splat generation"
```

---

### Task 4: SHARP 360 Pipeline

**Files:**
- Create: `spag4d/sharp360.py`
- Create: `tests/test_sharp360.py`

- [ ] **Step 1: Write failing tests for SHARP 360 pipeline helpers**

```python
# tests/test_sharp360.py
"""Tests for SHARP 360 pipeline helper functions."""

import numpy as np
import pytest


class TestFaceOrientation:
    def test_front_rotation_matrix(self):
        from spag4d.sharp360 import make_horizon_view
        view = make_horizon_view(0, 4)
        assert view.name == "front"
        R = view.rotation_matrix
        assert R.shape == (3, 3)
        # Front view: forward should be (0, 0, 1)
        np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-5)

    def test_right_view(self):
        from spag4d.sharp360 import make_horizon_view
        view = make_horizon_view(1, 4)
        assert view.name == "right"

    def test_six_sides_naming(self):
        from spag4d.sharp360 import make_horizon_view
        view = make_horizon_view(0, 6)
        assert view.name == "side_01"
        view5 = make_horizon_view(5, 6)
        assert view5.name == "side_06"


class TestExtractionLayout:
    def test_build_layout_4_sides(self):
        from spag4d.sharp360 import build_extraction_layout
        layout = build_extraction_layout(
            face_size=1024, panorama_height=2048,
            side_count=4, overlap_degrees=10.0,
        )
        assert len(layout.views) == 4
        assert layout.image_width >= 1024
        assert layout.focal_px > 0

    def test_build_layout_6_sides(self):
        from spag4d.sharp360 import build_extraction_layout
        layout = build_extraction_layout(
            face_size=1365, panorama_height=4096,
            side_count=6, overlap_degrees=10.0,
        )
        assert len(layout.views) == 6


class TestExtractPerspectiveView:
    def test_output_shape(self):
        from spag4d.sharp360 import extract_perspective_view, make_horizon_view
        pano = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        view = make_horizon_view(0, 4)
        face = extract_perspective_view(
            pano, image_width=50, image_height=50,
            focal_x_px=40.0, focal_y_px=40.0, view=view,
        )
        assert face.shape == (50, 50, 3)
        assert face.dtype == np.uint8

    def test_uniform_panorama_gives_uniform_face(self):
        from spag4d.sharp360 import extract_perspective_view, make_horizon_view
        pano = np.full((100, 200, 3), 128, dtype=np.uint8)
        view = make_horizon_view(0, 4)
        face = extract_perspective_view(
            pano, image_width=50, image_height=50,
            focal_x_px=40.0, focal_y_px=40.0, view=view,
        )
        # Should be all 128 (uniform panorama)
        assert np.allclose(face, 128, atol=2)


class TestBilinearSample:
    def test_integer_coords_exact(self):
        from spag4d.sharp360 import bilinear_sample
        img = np.arange(12, dtype=np.uint8).reshape(3, 4, 1).repeat(3, axis=2)
        sx = np.array([[1.0]])
        sy = np.array([[1.0]])
        result = bilinear_sample(img, sx, sy)
        assert result.shape == (1, 1, 3)
        assert result[0, 0, 0] == img[1, 1, 0]

    def test_wraps_horizontally(self):
        from spag4d.sharp360 import bilinear_sample
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        img[:, 0, :] = 200
        img[:, 7, :] = 200
        # Sample at x=-0.5 should wrap to the right edge
        sx = np.array([[-0.5]])
        sy = np.array([[2.0]])
        result = bilinear_sample(img, sx, sy)
        assert result[0, 0, 0] > 100  # should be close to 200 (wrapped)


class TestBilinearSampleScalar:
    def test_integer_coords(self):
        from spag4d.sharp360 import bilinear_sample_scalar
        img = np.arange(12, dtype=np.float32).reshape(3, 4)
        sx = np.array([2.0])
        sy = np.array([1.0])
        result = bilinear_sample_scalar(img, sx, sy)
        assert result.shape == (1,)
        assert abs(result[0] - img[1, 2]) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sharp360.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spag4d.sharp360'`

- [ ] **Step 3: Implement SHARP 360 pipeline**

```python
# spag4d/sharp360.py
"""SHARP 360 pipeline: per-face SHARP prediction with DA360 alignment.

Extracts perspective views from a 360 panorama, runs Apple's SHARP model
on each face to predict Gaussians directly, aligns depths using DA360,
rotates into world frame, and merges into a single PLY.
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Ensure vendored ml-sharp is importable
_SHARP_SRC = str(Path(__file__).resolve().parent / "sharp_arch" / "ml-sharp" / "src")
if _SHARP_SRC not in sys.path:
    sys.path.insert(0, _SHARP_SRC)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceOrientation:
    """Perspective view orientation defined by right/down/forward vectors."""
    name: str
    right: tuple
    down: tuple
    forward: tuple

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.column_stack((self.right, self.down, self.forward)).astype(np.float32)


@dataclass(frozen=True)
class ExtractionLayout:
    """Layout of N perspective views extracted from a panorama."""
    name: str
    views: tuple
    focal_px: float
    focal_y_px: float
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# View geometry
# ---------------------------------------------------------------------------

def make_horizon_view(index: int, side_count: int) -> FaceOrientation:
    """Create a horizon-looking perspective view at a given azimuth index."""
    yaw = (2.0 * np.pi * index) / side_count
    forward = (float(np.sin(yaw)), 0.0, float(np.cos(yaw)))
    right = (float(np.cos(yaw)), 0.0, float(-np.sin(yaw)))
    down = (0.0, 1.0, 0.0)

    if side_count == 2:
        name = ("front", "back")[index]
    elif side_count == 4:
        name = ("front", "right", "back", "left")[index]
    else:
        name = f"side_{index + 1:02d}"

    return FaceOrientation(name, right, down, forward)


def build_extraction_layout(
    face_size: int,
    panorama_height: int,
    side_count: int,
    overlap_degrees: float = 10.0,
) -> ExtractionLayout:
    """Build the perspective extraction layout for N horizon views."""
    span_degrees = 360.0 / side_count
    view_fov_degrees = min(170.0, span_degrees + overlap_degrees)
    image_width = max(face_size, int(round(face_size * (view_fov_degrees / span_degrees))))
    focal_px = (image_width / 2.0) / np.tan(np.deg2rad(view_fov_degrees) / 2.0)
    image_height = panorama_height
    focal_y_px = focal_px  # square pixels
    views = tuple(make_horizon_view(i, side_count) for i in range(side_count))
    return ExtractionLayout(
        f"horizon{side_count}", views, focal_px, focal_y_px, image_width, image_height,
    )


# ---------------------------------------------------------------------------
# Perspective extraction
# ---------------------------------------------------------------------------

def bilinear_sample(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    """Bilinear sample from an RGB image with horizontal wrapping."""
    height, width, channels = image.shape
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = x0 % width
    y0 = np.clip(y0, 0, height - 1)
    wx = (sample_x - np.floor(sample_x))[..., None]
    wy = (sample_y - np.floor(sample_y))[..., None]
    img_f = image.astype(np.float32)
    top = img_f[y0, x0] * (1.0 - wx) + img_f[y0, x1] * wx
    bottom = img_f[y1, x0] * (1.0 - wx) + img_f[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8).reshape(sample_x.shape + (channels,))


def bilinear_sample_scalar(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    """Bilinear sample from a scalar (H, W) image with horizontal wrapping."""
    height, width = image.shape
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = x0 % width
    y0 = np.clip(y0, 0, height - 1)
    wx = sample_x - np.floor(sample_x)
    wy = sample_y - np.floor(sample_y)
    img_f = image.astype(np.float32)
    top = img_f[y0, x0] * (1.0 - wx) + img_f[y0, x1] * wx
    bottom = img_f[y1, x0] * (1.0 - wx) + img_f[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def extract_perspective_view(
    panorama: np.ndarray,
    image_width: int,
    image_height: int,
    focal_x_px: float,
    focal_y_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    """Extract a perspective view from an equirectangular panorama."""
    cx = np.arange(image_width, dtype=np.float32) + 0.5
    cy = np.arange(image_height, dtype=np.float32) + 0.5
    cx = (cx - image_width / 2.0) / focal_x_px
    cy = (cy - image_height / 2.0) / focal_y_px
    gx, gy = np.meshgrid(cx, cy)
    local_dirs = np.stack((gx, gy, np.ones_like(gx)), axis=-1)
    local_dirs /= np.linalg.norm(local_dirs, axis=-1, keepdims=True)
    R = view.rotation_matrix
    world_dirs = local_dirs @ R.T
    wx, wy, wz = world_dirs[..., 0], np.clip(world_dirs[..., 1], -1.0, 1.0), world_dirs[..., 2]
    h, w = panorama.shape[:2]
    lon = np.arctan2(wx, wz)
    lat = np.arcsin(wy)
    sx = (lon / (2.0 * np.pi) + 0.5) * w - 0.5
    sy = (lat / np.pi + 0.5) * h - 0.5
    return bilinear_sample(panorama, sx, sy)


def extract_perspective_views(
    layout: ExtractionLayout, panorama: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract all perspective views defined by the layout."""
    return {
        v.name: extract_perspective_view(
            panorama, layout.image_width, layout.image_height,
            layout.focal_px, layout.focal_y_px, v,
        )
        for v in layout.views
    }


def extract_perspective_scalar_view(
    panorama: np.ndarray,
    image_width: int, image_height: int,
    focal_x_px: float, focal_y_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    """Extract a scalar (depth/disparity) perspective view from an ERP map."""
    cx = np.arange(image_width, dtype=np.float32) + 0.5
    cy = np.arange(image_height, dtype=np.float32) + 0.5
    cx = (cx - image_width / 2.0) / focal_x_px
    cy = (cy - image_height / 2.0) / focal_y_px
    gx, gy = np.meshgrid(cx, cy)
    local_dirs = np.stack((gx, gy, np.ones_like(gx)), axis=-1)
    local_dirs /= np.linalg.norm(local_dirs, axis=-1, keepdims=True)
    R = view.rotation_matrix
    world_dirs = local_dirs @ R.T
    wx, wy, wz = world_dirs[..., 0], np.clip(world_dirs[..., 1], -1.0, 1.0), world_dirs[..., 2]
    h, w = panorama.shape[:2]
    lon = np.arctan2(wx, wz)
    lat = np.arcsin(wy)
    sx = (lon / (2.0 * np.pi) + 0.5) * w - 0.5
    sy = (lat / np.pi + 0.5) * h - 0.5
    return bilinear_sample_scalar(panorama, sx, sy)


# ---------------------------------------------------------------------------
# Gaussian clipping and alignment
# ---------------------------------------------------------------------------

def filter_gaussians_by_view_border(gaussians, clip_degrees: float):
    """Clip Gaussians to within clip_degrees of the view center (hard Voronoi)."""
    from sharp.utils.gaussians import Gaussians3D

    if clip_degrees >= 179.0:
        return gaussians
    half_rad = np.deg2rad(clip_degrees / 2.0)
    h_limit = float(np.tan(half_rad))
    mv = gaussians.mean_vectors
    depth = mv[..., 2]
    h_ratio = torch.abs(mv[..., 0]) / torch.clamp(depth, min=1e-6)
    keep = (depth > 0.0) & (h_ratio <= h_limit)
    if int(keep.sum()) == 0:
        raise ValueError("View-border clipping removed all Gaussians.")
    if int(keep.sum()) == int(keep.numel()):
        return gaussians
    m = keep[0]
    return Gaussians3D(
        mean_vectors=mv[:, m, :],
        singular_values=gaussians.singular_values[:, m, :],
        quaternions=gaussians.quaternions[:, m, :],
        colors=gaussians.colors[:, m, ...],
        opacities=gaussians.opacities[:, m, ...],
    )


def align_gaussians_to_reference(
    gaussians,
    reference_disparity_view: np.ndarray,
    focal_x_px: float, focal_y_px: float,
    image_width: int, image_height: int,
    grid_resolution: int = 8,
):
    """Align Gaussian depths to DA360 disparity using a smooth grid scale field.

    Returns (aligned_gaussians, median_scale, sample_count).
    """
    from sharp.utils.gaussians import Gaussians3D

    grid_cx = max(1, int(grid_resolution))
    grid_cy = max(1, int(round(grid_cx * (image_height / max(1, image_width)))))

    mv = gaussians.mean_vectors
    mv_np = mv[0].detach().cpu().numpy().astype(np.float32)
    dz = mv_np[:, 2]
    radial = np.linalg.norm(mv_np, axis=1)

    valid = dz > 1e-6
    px = (mv_np[:, 0] / np.clip(dz, 1e-6, None)) * focal_x_px + (image_width / 2.0) - 0.5
    py = (mv_np[:, 1] / np.clip(dz, 1e-6, None)) * focal_y_px + (image_height / 2.0) - 0.5
    valid &= (px >= 0) & (px <= image_width - 1) & (py >= 0) & (py <= image_height - 1)

    per_point_scale = np.ones(mv_np.shape[0], dtype=np.float32)
    median_scale = 1.0
    count = 0

    if int(valid.sum()) >= 64:
        ref_disp = bilinear_sample_scalar(reference_disparity_view, px[valid], py[valid])
        ok = np.isfinite(ref_disp) & (ref_disp > 1e-6) & (radial[valid] > 1e-6)
        count = int(ok.sum())
        if count >= 64:
            ref_depth = (1.0 / ref_disp[ok]).astype(np.float32)
            sharp_r = radial[valid][ok]
            raw_scale = ref_depth / sharp_r

            lo, hi = np.quantile(raw_scale, [0.05, 0.95])
            trimmed = raw_scale[(raw_scale >= lo) & (raw_scale <= hi)]
            median_scale = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(raw_scale))

            # Build coarse NxN grid of median scales
            px_ok, py_ok = px[valid][ok], py[valid][ok]
            cw, ch = image_width / grid_cx, image_height / grid_cy
            grid = np.full((grid_cy, grid_cx), median_scale, dtype=np.float32)
            for gy in range(grid_cy):
                for gx in range(grid_cx):
                    in_cell = (
                        (px_ok >= gx * cw) & (px_ok < (gx + 1) * cw) &
                        (py_ok >= gy * ch) & (py_ok < (gy + 1) * ch)
                    )
                    if int(in_cell.sum()) >= 8:
                        cs = raw_scale[in_cell]
                        cl, ch_ = np.quantile(cs, [0.1, 0.9])
                        ct = cs[(cs >= cl) & (cs <= ch_)]
                        if ct.size > 0:
                            grid[gy, gx] = float(np.median(ct))
            grid = np.clip(grid, median_scale * 0.1, median_scale * 10.0)

            # Bilinear interpolate grid to every Gaussian
            all_px = np.clip(px, 0, image_width - 1)
            all_py = np.clip(py, 0, image_height - 1)
            gxc = all_px / cw - 0.5
            gyc = all_py / ch - 0.5
            gx0 = np.clip(np.floor(gxc).astype(np.int32), 0, grid_cx - 1)
            gy0 = np.clip(np.floor(gyc).astype(np.int32), 0, grid_cy - 1)
            gx1 = np.clip(gx0 + 1, 0, grid_cx - 1)
            gy1 = np.clip(gy0 + 1, 0, grid_cy - 1)
            wx = np.clip(gxc - gx0, 0, 1).astype(np.float32)
            wy = np.clip(gyc - gy0, 0, 1).astype(np.float32)
            per_point_scale = (
                grid[gy0, gx0] * (1 - wx) * (1 - wy) +
                grid[gy0, gx1] * wx * (1 - wy) +
                grid[gy1, gx0] * (1 - wx) * wy +
                grid[gy1, gx1] * wx * wy
            )
            per_point_scale[~valid] = median_scale

    device, dtype = mv.device, mv.dtype
    scale_t = torch.from_numpy(per_point_scale).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)

    return Gaussians3D(
        mean_vectors=mv * scale_t,
        singular_values=gaussians.singular_values * scale_t,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    ), median_scale, count


def scale_gaussians(gaussians, scale_factor: float):
    """Uniformly scale Gaussian positions and sizes."""
    from sharp.utils.gaussians import Gaussians3D
    if abs(scale_factor - 1.0) < 1e-5:
        return gaussians
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors * scale_factor,
        singular_values=gaussians.singular_values * scale_factor,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def merge_gaussians(gaussians_list):
    """Concatenate multiple Gaussians3D into one."""
    from sharp.utils.gaussians import Gaussians3D
    if not gaussians_list:
        raise ValueError("No face Gaussians to merge.")
    return Gaussians3D(
        mean_vectors=torch.cat([g.mean_vectors for g in gaussians_list], dim=1),
        singular_values=torch.cat([g.singular_values for g in gaussians_list], dim=1),
        quaternions=torch.cat([g.quaternions for g in gaussians_list], dim=1),
        colors=torch.cat([g.colors for g in gaussians_list], dim=1),
        opacities=torch.cat([g.opacities for g in gaussians_list], dim=1),
    )


# ---------------------------------------------------------------------------
# DA360 depth inference (reuses existing DA360 model)
# ---------------------------------------------------------------------------

def predict_da360_disparity(panorama: np.ndarray, device: torch.device) -> np.ndarray:
    """Run DA360 on a panorama and return (H, W) disparity map.

    Uses the existing DA360 model from spag4d.da360_model.
    """
    from .da360_model import DA360Model

    model = DA360Model.load(device=device)
    image_tensor = torch.from_numpy(panorama).to(device)

    with torch.inference_mode():
        depth, _ = model.predict(image_tensor)

    # DA360 returns depth; convert to disparity for alignment
    disparity = 1.0 / torch.clamp(depth, min=1e-6)
    return disparity.cpu().numpy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def convert_sharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    side_count: int = 6,
    overlap_degrees: float = 10.0,
    seedvr2_upscale: bool = False,
    seedvr2_config=None,
    progress_callback=None,
) -> dict:
    """Run the full SHARP 360 pipeline.

    Args:
        input_path: Path to equirectangular panorama.
        output_path: Path for output PLY file.
        device: Torch device.
        side_count: Number of horizon views (default 6).
        overlap_degrees: Overlap between adjacent views in degrees.
        seedvr2_upscale: Whether to upscale faces with SeedVR2 before SHARP.
        seedvr2_config: SeedVR2Config for upscaling (required if seedvr2_upscale=True).
        progress_callback: Optional (stage: str, pct: int) callback.

    Returns:
        Dict with output_path, splat_count, processing_time.
    """
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import apply_transform, save_ply

    start_time = time.time()

    def report(stage, pct):
        logger.info(f"[sharp360] {stage} ({pct}%%)")
        if progress_callback:
            progress_callback(stage, pct)

    # Step 1: Load panorama
    report("load", 0)
    from PIL import Image, ImageOps
    img = Image.open(input_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    panorama = np.asarray(img).copy()
    h, w = panorama.shape[:2]
    if w != h * 2:
        raise ValueError(f"Expected 2:1 ERP panorama, got {w}x{h}")
    logger.info(f"[sharp360] Loaded panorama {w}x{h}")

    # Step 2: Build extraction layout
    report("layout", 5)
    face_size = max(256, w // side_count)
    layout = build_extraction_layout(face_size, h, side_count, overlap_degrees)
    logger.info(f"[sharp360] Layout: {side_count} views, {layout.image_width}x{layout.image_height}, "
                f"focal={layout.focal_px:.1f}")

    # Step 3: Extract perspective faces
    report("extract", 10)
    faces = extract_perspective_views(layout, panorama)
    image_width = layout.image_width
    image_height = layout.image_height
    focal_px = layout.focal_px
    focal_y_px = layout.focal_y_px

    # Step 4: Optional SeedVR2 upscale
    if seedvr2_upscale and seedvr2_config is not None:
        report("seedvr2_upscale", 15)
        from .seedvr2 import upscale_images
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            original_width = image_width
            faces, image_width, image_height = upscale_images(
                faces, seedvr2_config, Path(tmp),
            )
            scale_ratio = image_width / original_width
            focal_px *= scale_ratio
            focal_y_px *= scale_ratio
            logger.info(f"[sharp360] SeedVR2 upscaled to {image_width}x{image_height}")

    # Step 5: DA360 depth for alignment
    report("da360", 25)
    ref_disparity_pano = predict_da360_disparity(panorama, device)
    ref_disparity_views = {
        v.name: extract_perspective_scalar_view(
            ref_disparity_pano, image_width, image_height,
            focal_px, focal_y_px, v,
        )
        for v in layout.views
    }
    # Free DA360 VRAM before loading SHARP
    torch.cuda.empty_cache()

    # Step 6: SHARP prediction per face
    report("sharp_load", 35)
    logger.info("[sharp360] Loading SHARP predictor...")
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    from sharp.models import PredictorParams, create_predictor
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)

    raw_face_gaussians = {}
    for i, view in enumerate(layout.views):
        pct = 40 + int(30 * i / len(layout.views))
        report(f"sharp_{view.name}", pct)
        logger.info(f"[sharp360] SHARP predict: {view.name}")
        gaussians = predict_image(predictor, faces[view.name], (focal_px, focal_y_px), device)
        raw_face_gaussians[view.name] = gaussians.to(torch.device("cpu"))

    del predictor
    torch.cuda.empty_cache()

    # Step 7: Clip to view borders
    report("clip", 72)
    clip_degrees = 360.0 / side_count
    clipped = {}
    original_median_radii = []
    for view in layout.views:
        clipped[view.name] = filter_gaussians_by_view_border(
            raw_face_gaussians[view.name], clip_degrees,
        )
        original_median_radii.append(float(torch.median(
            torch.norm(clipped[view.name].mean_vectors, dim=-1),
        ).item()))

    # Step 8: DA360 alignment
    report("align", 78)
    aligned = {}
    for view in layout.views:
        g = clipped[view.name].to(device)
        g, med_scale, count = align_gaussians_to_reference(
            g, ref_disparity_views[view.name],
            focal_px, focal_y_px, image_width, image_height,
        )
        aligned[view.name] = g
        logger.info(f"[sharp360] Aligned {view.name}: scale={med_scale:.4f} ({count} samples)")

    # Step 9: Rotate + merge
    report("merge", 85)
    rotated_list = []
    for view in layout.views:
        transform = torch.eye(3, 4, dtype=torch.float32, device=device)
        transform[:, :3] = torch.from_numpy(view.rotation_matrix).to(device=device, dtype=torch.float32)
        rotated = apply_transform(aligned[view.name], transform).to(torch.device("cpu"))
        rotated_list.append(rotated)
    merged = merge_gaussians(rotated_list)

    # Global scale restore
    original_scene_median = float(np.median(original_median_radii))
    current_median = float(torch.median(torch.norm(merged.mean_vectors, dim=-1)).item())
    if current_median > 1e-8:
        restore_scale = original_scene_median / current_median
        merged = scale_gaussians(merged, restore_scale)
        logger.info(f"[sharp360] Global scale restore: {restore_scale:.4f}")

    # Step 10: Save PLY
    report("save", 92)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_ply(merged, (focal_px, focal_y_px), (image_height, image_width), Path(output_path))

    splat_count = int(merged.mean_vectors.shape[1])
    elapsed = time.time() - start_time
    logger.info(f"[sharp360] Done: {splat_count:,} Gaussians in {elapsed:.1f}s")

    report("done", 100)
    return {
        "output_path": output_path,
        "splat_count": splat_count,
        "processing_time": elapsed,
        "side_count": side_count,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sharp360.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/sharp360.py tests/test_sharp360.py
git commit -m "feat: add SHARP 360 pipeline (per-face prediction + DA360 alignment + merge)"
```

---

### Task 5: Generator Dispatch in core.py

**Files:**
- Modify: `spag4d/core.py`

- [ ] **Step 1: Add generator dispatch to SPAG4D.convert()**

In `spag4d/core.py`, add `generator` parameter to `__init__` and dispatch in `convert()`.

Add to `__init__` signature (line 33-38):

```python
    def __init__(
        self,
        device: str = "cuda",
        depth_model: str = "da360",
        model_path: Optional[str] = None,
        use_mock_dap: bool = False,
        generator: Optional[str] = None,
    ):
```

Add after line 47 (`self._depth_models = {}`):

```python
        # Generator: "da360", "dap", or "sharp360"
        # If generator is set, it overrides depth_model for the dispatch
        self.generator = generator or depth_model
```

Only eagerly load depth model for SPAG paths (replace line 49):

```python
        if self.generator in ("da360", "dap"):
            self._get_depth_model(self.generator)
```

Add `generator` parameter to `convert()` method signature (after `depth_npy_path` at line 85):

```python
        generator: Optional[str] = None,
        side_count: int = 6,
        seedvr2_upscale: bool = False,
```

Add dispatch at the top of `convert()` body, right after `image_tensor = ...` (after line 127):

```python
        # Dispatch to SHARP 360 if requested
        active_generator = generator or self.generator
        if active_generator == "sharp360":
            from .sharp360 import convert_sharp360
            from .seedvr2 import SeedVR2Config

            seedvr2_cfg = SeedVR2Config() if seedvr2_upscale else None
            result_dict = convert_sharp360(
                input_path=str(input_path),
                output_path=str(output_path),
                device=self.device,
                side_count=side_count,
                seedvr2_upscale=seedvr2_upscale,
                seedvr2_config=seedvr2_cfg,
            )
            file_size = Path(output_path).stat().st_size
            return ConversionResult(
                output_path=str(output_path),
                splat_count=result_dict["splat_count"],
                file_size=file_size,
                processing_time=result_dict["processing_time"],
                depth_range=(0.0, 0.0),  # SHARP doesn't use depth range
                panorama_size=(W, H),
            )
```

- [ ] **Step 2: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add spag4d/core.py
git commit -m "feat: add generator dispatch — sharp360 alongside da360/dap"
```

---

### Task 6: CLI Updates

**Files:**
- Modify: `spag4d/cli.py`

- [ ] **Step 1: Add --generator, --side-count, --seedvr2-upscale CLI options**

Add new options to the `convert` command (after the `--depth-model` option at line 20):

```python
@click.option('--generator', type=click.Choice(['da360', 'dap', 'sharp360']),
              default=None, help='Splat generator (default: da360). sharp360 uses Apple SHARP.')
@click.option('--side-count', type=int, default=6,
              help='Number of horizon views for SHARP 360 (default: 6)')
@click.option('--seedvr2-upscale', is_flag=True,
              help='Upscale SHARP face images with SeedVR2 before prediction')
```

Add corresponding parameters to the `convert` function signature:

```python
    generator: str,
    side_count: int,
    seedvr2_upscale: bool,
```

Update the mode display (line 83):

```python
    active_gen = generator or depth_model
    if active_gen == "sharp360":
        mode = f"SHARP 360 ({side_count} views" + (", SeedVR2 upscale" if seedvr2_upscale else "") + ")"
    elif sharp_refine:
        mode = "SHARP refined"
    else:
        mode = f"SPAG (stride={stride})"
    click.echo(f"Loading SPAG-4D [{active_gen.upper()} + {mode}]...")
```

Pass generator params to `converter.convert()` in `run_single` (line 96):

```python
    def run_single(img_path, out_path):
        return converter.convert(
            input_path=str(img_path),
            output_path=str(out_path),
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            stride=stride,
            outlier_pruning=outlier_pruning,
            global_scale=global_scale,
            force_erp=force_erp,
            generator=generator,
            side_count=side_count,
            seedvr2_upscale=seedvr2_upscale,
        )
```

Add `sharp` and `seedvr2` to the `download-models` command choices (line 167):

```python
@click.option('--model', type=click.Choice(['dap', 'da360', 'gsfix3d', 'sharp', 'seedvr2', 'all']),
              default='all', help='Which model weights to download')
```

Add download handlers for sharp and seedvr2 before the final closing of the function:

```python
    if model in ('sharp', 'all'):
        click.echo("SHARP checkpoint auto-downloads on first use via torch.hub.")
        click.echo("No manual download needed.")

    if model in ('seedvr2', 'all'):
        click.echo("SeedVR2 requires manual installation:")
        click.echo("  1. git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git third_party/seedvr2_videoupscaler")
        click.echo("  2. Download model weights into third_party/seedvr2_videoupscaler/models/")
```

- [ ] **Step 2: Test CLI help**

Run: `.venv/Scripts/python.exe -m spag4d convert --help`
Expected: Shows `--generator`, `--side-count`, `--seedvr2-upscale` options

- [ ] **Step 3: Commit**

```bash
git add spag4d/cli.py
git commit -m "feat: add --generator sharp360, --side-count, --seedvr2-upscale CLI options"
```

---

### Task 7: API Updates

**Files:**
- Modify: `api.py`

- [ ] **Step 1: Add generator parameter to /api/convert endpoint**

In `api.py`, update the `convert_panorama` function signature (line 204-216). Add after the `depth_model` param:

```python
    generator: Optional[str] = Query(None, pattern="^(da360|dap|sharp360)$"),
    side_count: int = Query(6, ge=2, le=12),
    seedvr2_upscale: bool = Query(False),
```

Add to `job.params` dict (after line 248):

```python
        "generator": generator,
        "side_count": side_count,
        "seedvr2_upscale": seedvr2_upscale,
```

Pass to `process_job` (add to the `asyncio.create_task` call, line 259):

```python
        generator=generator,
        side_count=side_count,
        seedvr2_upscale=seedvr2_upscale,
```

Update `process_job` signature (line 279) to add:

```python
    generator: Optional[str] = None,
    side_count: int = 6,
    seedvr2_upscale: bool = False,
```

Add `generator`, `side_count`, `seedvr2_upscale` to the `processor.convert()` call (line 298-313):

```python
                generator=generator,
                side_count=side_count,
                seedvr2_upscale=seedvr2_upscale,
```

- [ ] **Step 2: Test endpoint**

Run: `.venv/Scripts/python.exe -m spag4d serve --port 7860` then test with:
```bash
curl -X POST "http://localhost:7860/api/convert?generator=da360&stride=2" -F file=@test.jpg
```
Expected: Normal conversion works (generator=da360 is equivalent to depth_model=da360)

- [ ] **Step 3: Commit**

```bash
git add api.py
git commit -m "feat: add generator param to /api/convert endpoint"
```

---

### Task 8: UI Updates

**Files:**
- Modify: `static/index.html`
- Modify: `static/js/app.js`

- [ ] **Step 1: Update index.html — generator dropdown and SHARP settings**

Replace the depth-model dropdown section (lines 173-177) with:

```html
                        <div class="field" style="min-width: 100px;"
                        title="Splat generator. DA360/DAP = depth-based SPAG. SHARP 360 = per-face ML prediction.">
                        <label for="generator">Generator</label>
                        <select id="generator" style="width: 120px;">
                            <option value="da360" selected>DA360</option>
                            <option value="dap">DAP</option>
                            <option value="sharp360">SHARP 360</option>
                        </select>
                    </div>
```

Add SHARP-specific settings (hidden by default) after the DEPTH SCALE field:

```html
                    <!-- SHARP 360 settings (shown when generator=sharp360) -->
                    <div id="sharp360-settings" style="display:none;">
                        <div class="field" style="min-width: 80px;">
                            <label for="side-count">Sides</label>
                            <select id="side-count" style="width: 60px;">
                                <option value="4">4</option>
                                <option value="6" selected>6</option>
                                <option value="8">8</option>
                                <option value="10">10</option>
                                <option value="12">12</option>
                            </select>
                        </div>
                        <div class="field" style="min-width: 100px;">
                            <label for="seedvr2-upscale">SeedVR2 Upscale</label>
                            <input type="checkbox" id="seedvr2-upscale">
                        </div>
                    </div>
```

- [ ] **Step 2: Update app.js — generator routing and conditional UI**

Add to constructor (`this.depthModelInput` replacement around line 30):

```javascript
        this.generatorInput = document.getElementById('generator');
        this.sideCountInput = document.getElementById('side-count');
        this.seedvr2UpscaleInput = document.getElementById('seedvr2-upscale');
        this.sharp360Settings = document.getElementById('sharp360-settings');
        this.spagSettings = document.getElementById('spag-settings'); // we'll wrap existing SPAG fields

        // Show/hide SHARP settings based on generator selection
        this.generatorInput.addEventListener('change', () => this.updateGeneratorUI());
```

Add the `updateGeneratorUI` method:

```javascript
    updateGeneratorUI() {
        const isSharp = this.generatorInput.value === 'sharp360';
        this.sharp360Settings.style.display = isSharp ? '' : 'none';
        // Hide stride/depth controls for SHARP (not applicable)
        const spagFields = document.querySelectorAll('.spag-only');
        spagFields.forEach(el => el.style.display = isSharp ? 'none' : '');
    }
```

Update the convert params (around line 229) to use generator:

```javascript
            const generator = this.generatorInput.value;
            const params = new URLSearchParams({
                generator: generator,
                stride: this.strideInput.value,
                depth_min: this.depthMinInput.value,
                depth_max: this.depthMaxInput.value,
                // ... existing params ...
            });

            // Add SHARP-specific params
            if (generator === 'sharp360') {
                params.set('side_count', this.sideCountInput.value);
                params.set('seedvr2_upscale', this.seedvr2UpscaleInput.checked);
            }

            // Keep depth_model for backward compat
            if (generator !== 'sharp360') {
                params.set('depth_model', generator);
            }
```

- [ ] **Step 3: Test UI**

Open `http://localhost:7860`, verify:
1. Generator dropdown shows DA360, DAP, SHARP 360
2. Selecting SHARP 360 shows side count + SeedVR2 checkbox, hides stride/depth fields
3. Selecting DA360/DAP shows normal SPAG controls
4. DA360 conversion still works

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/js/app.js
git commit -m "feat: add SHARP 360 generator option to web UI"
```

---

### Task 9: Update Tests and Clean Up

**Files:**
- Modify: `tests/test_omniroam_config.py`
- Modify: `spag4d/refine/__init__.py`

- [ ] **Step 1: Update omniroam_config test for removed fields**

Remove or update any tests that reference `seedvr2_install_dir` or `seedvr2_batch_size` in `tests/test_omniroam_config.py`. The `test_seedvr2_defaults` test should check the remaining native fields:

```python
def test_seedvr2_defaults():
    cfg = OmniRoamConfig()
    assert cfg.upscale_backend == "none"
    assert cfg.seedvr2_model == "seedvr2_ema_3b_fp16.safetensors"
    assert cfg.seedvr2_target_resolution == 1024
    assert cfg.seedvr2_color_correction == "lab"
    assert cfg.seedvr2_block_swap == 0
```

- [ ] **Step 2: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests PASS, no references to deleted WSL2 seedvr2 adapter

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: update tests for native SeedVR2 and SHARP 360 integration"
```

---

## Self-Review

**Spec coverage check:**

| Spec Section | Task(s) |
|---|---|
| 1. Generator Selection | Task 5 (core.py dispatch) |
| 2. SHARP 360 Pipeline | Task 4 (sharp360.py) |
| 3. Native SeedVR2 Adapter | Task 1 (seedvr2.py) |
| 4. Integration Points | Tasks 5-8 (core, cli, api, UI) |
| 5. Files Changed | Tasks 1-3 (create/remove/modify) |
| 6. Model Weights | Task 6 (download-models CLI) |
| 7. VRAM Requirements | Task 4 (sequential model loading in pipeline) |
| 8. License | Task 3 (vendored with license files) |

**Placeholder scan:** No TBD, TODO, or "fill in" patterns found. All steps have code.

**Type consistency:** `SeedVR2Config` used consistently across Task 1, 2, 4, 5. `FaceOrientation`, `ExtractionLayout` defined in Task 4 and used throughout. `convert_sharp360()` return dict matches the `ConversionResult` mapping in Task 5.
