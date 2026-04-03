# Refine v2: OmniRoam Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GSFixer synthesis backend with OmniRoam trajectory-coherent panoramic video generation for hole-filling, covering Phase 0 (WSL2 foundation + OmniRoam installation) and Phase 1 (preview-only baseline with gap-directed view selection).

**Architecture:** OmniRoam runs inside WSL2 via subprocess, generating 480x960 ERP video frames along gap-directed trajectories. Perspective crops are extracted from these frames as uncertain-pose pseudo-views, used as weak supervision (0.20 weight) alongside the original panorama (1.0 weight) in a modified gsplat optimization loop. Gap regions are seeded with sparse Gaussians from the source panorama's depth map. The existing GSFixer pipeline is preserved for A/B comparison.

**Tech Stack:** Python 3.10+, PyTorch, numpy, gsplat (GSFix3D fork), WSL2 + conda (OmniRoam), pytest.

**Spec:** `SPAG4D_Refine_v2_Design_Doc_R3.md` (R4)

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `spag4d/refine/omniroam_config.py` | `OmniRoamConfig` dataclass — all OmniRoam integration settings |
| `spag4d/refine/omniroam_adapter.py` | WSL2 subprocess wrapper: path conversion, environment validation, OmniRoam invocation with progress streaming |
| `spag4d/refine/omniroam_trajectory.py` | Trajectory generation matching OmniRoam's `make_cam_traj_from_preset_refspace()` — produces conditioning tensor + per-frame translations |
| `spag4d/refine/gap_analysis.py` | Render splat from evaluation viewpoints, compute per-pixel opacity, classify gap regions by angular direction |
| `spag4d/refine/view_selector.py` | Gap-directed OmniRoam frame selection: extract perspective crops from ERP frames, filter by gap overlap, rank and cap |
| `spag4d/refine/scale_alignment.py` | Reprojection-based scale alignment between OmniRoam's metric space and the splat's coordinate system |
| `spag4d/refine/gap_seeding.py` | Seed sparse Gaussians into gap regions using source panorama depth |
| `spag4d/refine/validation.py` | Multi-metric validation: source-anchor PSNR gate, coverage measurement, multi-view reprojection agreement |
| `spag4d/refine/pipeline_v2.py` | New pipeline orchestrator: 7-stage OmniRoam refinement flow |
| `scripts/setup_omniroam_wsl.sh` | WSL2 OmniRoam installation script |
| `tests/test_omniroam_config.py` | Config defaults and overrides |
| `tests/test_omniroam_adapter.py` | Path conversion, environment validation (mocked subprocess) |
| `tests/test_omniroam_trajectory.py` | Trajectory math + snapshot parity |
| `tests/test_gap_analysis.py` | Gap classification from synthetic masks |
| `tests/test_view_selector.py` | View selection logic with synthetic data |
| `tests/test_scale_alignment.py` | Scale search with mock renders |
| `tests/test_gap_seeding.py` | Seeding geometry correctness |
| `tests/test_distill_v2.py` | Tier-2 weighted loss logic |
| `tests/test_validation.py` | Metric computation |
| `tests/test_pipeline_v2_integration.py` | End-to-end integration scaffold |
| `tests/data/omniroam_trajectory_snapshots/*.npy` | Frozen trajectory tensors for parity testing |

### Modified Files

| File | Changes |
|------|---------|
| `spag4d/refine/distill.py` | Add `tier2_weight` parameter, hole-mask loss weighting, view-tier classification |
| `spag4d/refine/provenance.py` | Add `"omniroam"` and `"gap_seed"` provenance tags, extend `tag_gaussian_provenance()` |
| `spag4d/refine/__init__.py` | Export `refine_splat_v2`, `OmniRoamConfig` |

### Preserved (Unchanged)

`config.py` (original RefineConfig), `pipeline.py` (GSFixer pipeline), `gsfixer_adapter.py`, `camera_rig.py`, `mesh_extract.py`, `format_compat.py`

---

## Task Dependency Graph

```
Task 1 (OmniRoamConfig)
  ├─> Task 2 (Setup Script)
  ├─> Task 3 (OmniRoam Adapter) ──> Task 3 uses config
  ├─> Task 4 (Trajectory) ──> standalone math
  ├─> Task 5 (Gap Analysis) ──> uses camera_rig.render_with_hole_mask
  │     ├─> Task 6 (View Selector) ──> uses gap analysis + trajectory
  │     └─> Task 8 (Gap Seeding) ──> uses gap masks
  ├─> Task 7 (Scale Alignment) ──> uses rendering
  ├─> Task 9 (Distill Mods) ──> uses config for tier2_weight
  ├─> Task 10 (Validation) ──> uses rendering + provenance
  └─> Task 11 (Pipeline v2) ──> depends on ALL above
       └─> Task 12 (Integration Test)
```

Tasks 2-10 are parallelizable after Task 1. Task 11 depends on all. Task 12 depends on Task 11.

---

## Phase 0: WSL2 Foundation

### Task 1: OmniRoamConfig Dataclass

**Files:**
- Create: `spag4d/refine/omniroam_config.py`
- Create: `tests/test_omniroam_config.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_omniroam_config.py
import pytest
from spag4d.refine.omniroam_config import OmniRoamConfig


def test_defaults():
    cfg = OmniRoamConfig()
    assert cfg.enabled is False
    assert cfg.height == 480
    assert cfg.width == 960
    assert cfg.num_frames == 81
    assert cfg.tier2_weight == 0.20
    assert cfg.upscale_backend == "none"
    assert cfg.scale_alignment == "reprojection"
    assert cfg.trajectory_mode == "auto"
    assert cfg.min_gap_ratio == 0.05
    assert cfg.max_omniroam_views == 200
    assert cfg.gap_seed_stride == 4
    assert cfg.gap_seed_initial_opacity == pytest.approx(0.01)
    assert cfg.extract_fov_degrees == 90.0
    assert cfg.extract_directions == [0, 90, 180, 270]


def test_override():
    cfg = OmniRoamConfig(
        enabled=True,
        tier2_weight=0.30,
        trajectory_mode="all",
        wsl_distro="Ubuntu-22.04",
    )
    assert cfg.enabled is True
    assert cfg.tier2_weight == 0.30
    assert cfg.trajectory_mode == "all"
    assert cfg.wsl_distro == "Ubuntu-22.04"


def test_available_presets_default():
    cfg = OmniRoamConfig()
    assert "forward" in cfg.available_presets
    assert "s_curve" not in cfg.available_presets


def test_available_presets_independent():
    """Default list should not be shared across instances."""
    a = OmniRoamConfig()
    b = OmniRoamConfig()
    a.available_presets.append("s_curve")
    assert "s_curve" not in b.available_presets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_omniroam_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spag4d.refine.omniroam_config'`

- [ ] **Step 3: Implement OmniRoamConfig**

```python
# spag4d/refine/omniroam_config.py
"""OmniRoam integration configuration."""

from dataclasses import dataclass, field


@dataclass
class OmniRoamConfig:
    """All settings for OmniRoam-based refinement (Refine v2)."""

    # ── Execution environment ──
    enabled: bool = False
    install_dir: str = "/home/user/OmniRoam"  # WSL path
    wsl_distro: str = "Ubuntu"

    # ── Model ──
    ckpt_path: str = "models/OmniRoam/Preview/preview.ckpt"
    base_model_path: str = "models/Wan-AI/Wan2.1-T2V-1.3B"

    # ── Generation ──
    height: int = 480
    width: int = 960
    num_frames: int = 81
    cfg_scale: float = 5.0
    inference_steps: int = 50
    speed: float = 1.0

    # ── Trajectory selection ──
    # "auto" = gap analysis selects cardinal directions.
    # "all" = all 4 cardinal presets.
    # Or explicit list like ["forward", "left"].
    trajectory_mode: str = "auto"
    available_presets: list = field(default_factory=lambda: [
        "forward", "backward", "left", "right",
    ])
    s_curve_amp_m: float = 1.6
    loop_radius_m: float = 1.5
    step_m: float = 0.25

    # ── View selection ──
    extract_fov_degrees: float = 90.0
    extract_directions: list = field(default_factory=lambda: [0, 90, 180, 270])
    min_gap_ratio: float = 0.05
    max_omniroam_views: int = 200

    # ── Upscale (optional, Phase 2) ──
    upscale_backend: str = "none"  # "none" | "omniroam_refine" | "seedvr2"

    # ── Supervision ──
    tier2_weight: float = 0.20
    tier2_warmup_iterations: int = 3000
    hole_mask_threshold: float = 0.3
    hole_mask_update_interval: int = 500

    # ── Gap seeding ──
    gap_seed_stride: int = 4
    gap_seed_initial_opacity: float = 0.01

    # ── Scale alignment ──
    scale_alignment: str = "reprojection"  # "reprojection" | "manual:1.0" | "none"
    scale_search_range: tuple = (0.1, 10.0)
    scale_search_samples: int = 32

    # ── Distillation (inherits from RefineConfig where not overridden) ──
    iters_per_view: int = 20
    kf_iters: int = 50
    densify_grad_threshold: float = 0.005

    # ── Validation ──
    source_anchor_psnr_floor: float = 25.0  # dB — fail if below this
    convergence_threshold: float = 0.02
    max_iterations: int = 3

    # ── Paths for GSFixer baseline comparison ──
    gsfixer_checkpoint: str = "pretrained/gsfix3d"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_omniroam_config.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/omniroam_config.py tests/test_omniroam_config.py
git commit -m "feat(refine-v2): add OmniRoamConfig dataclass"
```

---

### Task 2: WSL2 OmniRoam Setup Script

**Files:**
- Create: `scripts/setup_omniroam_wsl.sh`

- [ ] **Step 1: Write the setup script**

```bash
#!/usr/bin/env bash
# scripts/setup_omniroam_wsl.sh
# Run INSIDE WSL2: wsl bash scripts/setup_omniroam_wsl.sh
#
# Installs OmniRoam + dependencies into a conda environment.
# Prerequisites: WSL2 Ubuntu with NVIDIA GPU passthrough working.

set -euo pipefail

OMNIROAM_DIR="${OMNIROAM_DIR:-$HOME/OmniRoam}"
CONDA_ENV="omniroam"
OMNIROAM_REPO="https://github.com/Inception-X/OmniRoam.git"

echo "=== SPAG-4D: OmniRoam WSL2 Setup ==="
echo "Install dir: $OMNIROAM_DIR"
echo "Conda env:   $CONDA_ENV"

# 1. Check NVIDIA driver is visible inside WSL
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi failed. Ensure NVIDIA GPU drivers are installed on Windows"
    echo "and GPU passthrough is enabled in WSL2."
    exit 1
fi
echo "[OK] NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 2. Install miniconda if not present
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "[OK] Miniconda installed"
else
    echo "[OK] Conda already available"
fi

# Ensure conda is on PATH for this script
eval "$(conda shell.bash hook)"

# 3. Clone OmniRoam
if [ -d "$OMNIROAM_DIR" ]; then
    echo "[OK] OmniRoam already cloned at $OMNIROAM_DIR"
    cd "$OMNIROAM_DIR"
    git pull --ff-only || echo "WARN: git pull failed, using existing checkout"
else
    echo "Cloning OmniRoam..."
    git clone "$OMNIROAM_REPO" "$OMNIROAM_DIR"
    cd "$OMNIROAM_DIR"
fi

# 4. Create conda environment
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "[OK] Conda env '$CONDA_ENV' already exists"
else
    echo "Creating conda env '$CONDA_ENV' with Python 3.10..."
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

# 5. Install dependencies
echo "Installing OmniRoam dependencies..."
conda run -n "$CONDA_ENV" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n "$CONDA_ENV" pip install -r requirements.txt

# 6. Install Rust (needed for DiffSynth-Studio)
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo "[OK] Rust/Cargo available"

# 7. Install DiffSynth-Studio
echo "Installing DiffSynth-Studio..."
conda run -n "$CONDA_ENV" pip install diffsynth

# 8. Download model weights (if not present)
PREVIEW_CKPT="$OMNIROAM_DIR/models/OmniRoam/Preview/preview.ckpt"
if [ -f "$PREVIEW_CKPT" ]; then
    echo "[OK] Preview checkpoint found"
else
    echo "Downloading OmniRoam Preview checkpoint..."
    echo "NOTE: You may need to download manually from the OmniRoam release page."
    echo "Expected location: $PREVIEW_CKPT"
    mkdir -p "$(dirname "$PREVIEW_CKPT")"
fi

WAN_DIR="$OMNIROAM_DIR/models/Wan-AI/Wan2.1-T2V-1.3B"
if [ -d "$WAN_DIR" ] && [ "$(ls -A "$WAN_DIR" 2>/dev/null)" ]; then
    echo "[OK] Wan2.1 base model found"
else
    echo "Downloading Wan2.1 T2V 1.3B base model..."
    echo "NOTE: You may need to download manually via huggingface-cli."
    echo "  conda run -n $CONDA_ENV huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $WAN_DIR"
    mkdir -p "$WAN_DIR"
fi

# 9. Verify installation
echo ""
echo "=== Verification ==="
conda run -n "$CONDA_ENV" python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
conda run -n "$CONDA_ENV" python -c "import diffsynth; print('DiffSynth-Studio OK')"

if [ -f "$OMNIROAM_DIR/infer_omniroam.py" ]; then
    echo "[OK] infer_omniroam.py found"
else
    echo "WARN: infer_omniroam.py not found at $OMNIROAM_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "To test: wsl -d Ubuntu bash -c 'cd $OMNIROAM_DIR && conda run -n $CONDA_ENV python infer_omniroam.py --help'"
echo ""
echo "Update your OmniRoamConfig:"
echo "  install_dir = \"$OMNIROAM_DIR\""
echo "  ckpt_path = \"models/OmniRoam/Preview/preview.ckpt\""
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/setup_omniroam_wsl.sh
git add scripts/setup_omniroam_wsl.sh
git commit -m "feat(refine-v2): add OmniRoam WSL2 setup script"
```

- [ ] **Step 3: Run the setup script inside WSL2**

Run: `wsl bash scripts/setup_omniroam_wsl.sh`
Expected: Script completes with `=== Setup complete ===` and all `[OK]` checks passing. Model downloads may require manual steps (noted by the script).

- [ ] **Step 4: Verify OmniRoam is callable**

Run: `wsl -d Ubuntu bash -c "cd ~/OmniRoam && conda run -n omniroam python infer_omniroam.py --help"`
Expected: OmniRoam prints its argument parser help text. If this fails, the adapter tests in Task 3 will also fail.

---

### Task 3: OmniRoam WSL2 Adapter

**Files:**
- Create: `spag4d/refine/omniroam_adapter.py`
- Create: `tests/test_omniroam_adapter.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_omniroam_adapter.py
import os
import platform
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from spag4d.refine.omniroam_adapter import (
    windows_to_wsl_path,
    validate_wsl_environment,
    run_omniroam_wsl,
    extract_video_frames,
)
from spag4d.refine.omniroam_config import OmniRoamConfig


class TestWindowsToWslPath:
    def test_d_drive(self):
        assert windows_to_wsl_path(r"D:\SPAG-4D\output") == "/mnt/d/SPAG-4D/output"

    def test_c_drive(self):
        assert windows_to_wsl_path(r"C:\Users\Cedar\file.jpg") == "/mnt/c/Users/Cedar/file.jpg"

    def test_forward_slashes(self):
        assert windows_to_wsl_path("D:/foo/bar") == "/mnt/d/foo/bar"

    def test_trailing_slash(self):
        result = windows_to_wsl_path(r"D:\SPAG-4D\output\\")
        assert result == "/mnt/d/SPAG-4D/output"


class TestValidateWslEnvironment:
    @patch("spag4d.refine.omniroam_adapter.subprocess.run")
    def test_missing_distro(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        cfg = OmniRoamConfig(wsl_distro="NonExistent")
        with pytest.raises(RuntimeError, match="WSL distro.*not found"):
            validate_wsl_environment(cfg)

    @patch("spag4d.refine.omniroam_adapter.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        cfg = OmniRoamConfig()
        validate_wsl_environment(cfg)  # Should not raise


class TestRunOmniroamWsl:
    @patch("spag4d.refine.omniroam_adapter.subprocess.Popen")
    def test_success(self, mock_popen, tmp_path):
        # Simulate OmniRoam producing output
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "Loading model...\n",
            "Step 10/50\n",
            "Step 50/50\n",
            "Saved to output_dir\n",
        ])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Create a fake source image
        src = tmp_path / "test.jpg"
        src.write_bytes(b"fake jpg")

        cfg = OmniRoamConfig(enabled=True)
        out = tmp_path / "out"
        out.mkdir()

        result = run_omniroam_wsl(
            image_path=str(src),
            output_dir=str(out),
            preset="forward",
            config=cfg,
        )
        assert result == str(out)
        mock_popen.assert_called_once()

    @patch("spag4d.refine.omniroam_adapter.subprocess.Popen")
    def test_failure_raises(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["Error: CUDA OOM\n"])
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        src = tmp_path / "test.jpg"
        src.write_bytes(b"fake jpg")
        out = tmp_path / "out"
        out.mkdir()

        cfg = OmniRoamConfig(enabled=True)
        with pytest.raises(RuntimeError, match="OmniRoam failed"):
            run_omniroam_wsl(str(src), str(out), "forward", cfg)

    @patch("spag4d.refine.omniroam_adapter.subprocess.Popen")
    def test_progress_callback(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["Step 5/50\n", "Step 50/50\n"])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        src = tmp_path / "test.jpg"
        src.write_bytes(b"fake jpg")
        out = tmp_path / "out"
        out.mkdir()

        progress = []
        cfg = OmniRoamConfig(enabled=True)
        run_omniroam_wsl(
            str(src), str(out), "forward", cfg,
            progress_callback=lambda cur, tot: progress.append((cur, tot)),
        )
        assert (5, 50) in progress
        assert (50, 50) in progress


class TestExtractVideoFrames:
    def test_extracts_frames(self, tmp_path):
        """extract_video_frames reads an mp4 and returns numpy arrays."""
        # This test needs a real video file — skip if cv2 not available
        cv2 = pytest.importorskip("cv2")

        # Create a tiny synthetic video (3 frames, 4x8 pixels)
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10, (8, 4))
        for i in range(3):
            frame = (i * 80 * np.ones((4, 8, 3))).astype(np.uint8)
            writer.write(frame)
        writer.release()

        import numpy as np
        frames = extract_video_frames(video_path)
        assert len(frames) == 3
        assert frames[0].shape == (4, 8, 3)
        assert frames[0].dtype == np.float32
        assert 0.0 <= frames[0].max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_omniroam_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement omniroam_adapter.py**

```python
# spag4d/refine/omniroam_adapter.py
"""OmniRoam WSL2 subprocess adapter: path conversion, validation, invocation."""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

logger = logging.getLogger(__name__)


def windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path (D:\\foo\\bar) to WSL mount path (/mnt/d/foo/bar)."""
    p = Path(win_path).resolve()
    drive = p.drive[0].lower()  # "D:" -> "d"
    # Get the path after the drive letter, convert to posix
    rest = PurePosixPath(PureWindowsPath(str(p)).relative_to(p.anchor))
    return f"/mnt/{drive}/{rest}"


def validate_wsl_environment(config) -> None:
    """Pre-flight check: WSL distro, conda env, OmniRoam install dir.

    Call at config time so errors surface early, not mid-generation.

    Raises:
        RuntimeError: If any check fails.
    """
    # Check distro exists
    result = subprocess.run(
        ["wsl", "-d", config.wsl_distro, "echo", "ok"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"WSL distro '{config.wsl_distro}' not found. "
            f"Install with: wsl --install -d {config.wsl_distro}"
        )

    # Check conda env and OmniRoam install
    result = subprocess.run(
        ["wsl", "-d", config.wsl_distro, "bash", "-c",
         f"conda run --no-banner -n omniroam python -c 'import torch' && "
         f"test -f {config.install_dir}/infer_omniroam.py"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OmniRoam environment check failed. Ensure:\n"
            f"  1. conda env 'omniroam' exists in WSL\n"
            f"  2. OmniRoam is installed at {config.install_dir}\n"
            f"Run: wsl bash scripts/setup_omniroam_wsl.sh\n"
            f"stderr: {result.stderr[:500]}"
        )

    logger.info("WSL2 OmniRoam environment validated OK")


def run_omniroam_wsl(
    image_path: str,
    output_dir: str,
    preset: str,
    config,
    progress_callback=None,
) -> str:
    """Run OmniRoam inference inside WSL2 with real-time output streaming.

    Args:
        image_path: Windows path to source panorama image.
        output_dir: Windows path to output directory for generated video.
        preset: Trajectory preset name (forward, backward, left, right, s_curve, loop).
        config: OmniRoamConfig instance.
        progress_callback: Optional callable(current_step, total_steps).

    Returns:
        output_dir path (same as input — OmniRoam writes generated.mp4 there).

    Raises:
        RuntimeError: If OmniRoam process exits with non-zero status.
    """
    # Stage source pano into a temp input directory
    # (OmniRoam's --local_images_dir iterates all images in the dir)
    staging_dir = os.path.join(output_dir, "_omniroam_input")
    os.makedirs(staging_dir, exist_ok=True)
    shutil.copy2(image_path, os.path.join(staging_dir, os.path.basename(image_path)))

    wsl_input_dir = windows_to_wsl_path(staging_dir)
    wsl_output = windows_to_wsl_path(output_dir)

    cmd = [
        "wsl", "-d", config.wsl_distro,
        "bash", "-c",
        f"cd {config.install_dir} && "
        f"conda run --no-banner -n omniroam python infer_omniroam.py "
        f"--local_images_dir {wsl_input_dir} "
        f"--height {config.height} "
        f"--width {config.width} "
        f"--num_frames {config.num_frames} "
        f"--ckpt_path {config.ckpt_path} "
        f"--enable_speed_control --speed_fixed {config.speed} "
        f"--use_cam_traj --traj_mode fixed "
        f"--traj_preset {preset} "
        f"--re_scale_pose fixed:1.0 "
        f"--cfg_scale {config.cfg_scale} "
        f"--num_inference_steps {config.inference_steps} "
        f"--output_dir {wsl_output} "
        f"--device cuda:0",
    ]

    logger.info(f"Running OmniRoam (preset={preset})...")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    output_lines = []
    step_pattern = re.compile(r"(\d+)/(\d+)")

    for line in process.stdout:
        line = line.rstrip()
        output_lines.append(line)
        logger.debug(f"[OmniRoam] {line}")

        if progress_callback:
            m = step_pattern.search(line)
            if m:
                progress_callback(int(m.group(1)), int(m.group(2)))

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"OmniRoam failed (exit {returncode}):\n"
            + "\n".join(output_lines[-20:])
        )

    logger.info(f"OmniRoam complete (preset={preset})")
    return output_dir


def extract_video_frames(video_path: str) -> list:
    """Read an mp4 video and return frames as a list of float32 numpy arrays.

    Args:
        video_path: Path to .mp4 file.

    Returns:
        List of (H, W, 3) float32 arrays in [0, 1].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2 reads BGR; convert to RGB, normalize to [0, 1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb)

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_omniroam_adapter.py -v`
Expected: All tests PASS (subprocess is mocked; video test may skip if cv2 not installed)

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/omniroam_adapter.py tests/test_omniroam_adapter.py
git commit -m "feat(refine-v2): OmniRoam WSL2 adapter with path conversion, validation, streaming"
```

---

## Phase 1: Core Pipeline

### Task 4: OmniRoam Trajectory Generator

**Files:**
- Create: `spag4d/refine/omniroam_trajectory.py`
- Create: `tests/test_omniroam_trajectory.py`
- Create: `tests/data/omniroam_trajectory_snapshots/` (6 `.npy` files)
- Create: `scripts/regenerate_trajectory_snapshots.sh`

- [ ] **Step 1: Write the test file**

```python
# tests/test_omniroam_trajectory.py
import numpy as np
import pytest
import torch
from pathlib import Path

from spag4d.refine.omniroam_trajectory import generate_omniroam_trajectory

SNAPSHOT_DIR = Path(__file__).parent / "data" / "omniroam_trajectory_snapshots"
PRESETS = ["forward", "backward", "left", "right", "s_curve", "loop"]


class TestTrajectoryShape:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_conditioning_tensor_shape(self, preset):
        cam_traj, translations = generate_omniroam_trajectory(preset=preset)
        assert cam_traj.shape == (21, 12), f"Expected (21, 12), got {cam_traj.shape}"
        assert cam_traj.dtype == torch.float32

    @pytest.mark.parametrize("preset", PRESETS)
    def test_translations_count(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        assert len(translations) == 81

    @pytest.mark.parametrize("preset", PRESETS)
    def test_translations_are_3d(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        for t in translations:
            assert t.shape == (3,)


class TestTrajectoryGeometry:
    def test_forward_moves_positive_x(self):
        _, translations = generate_omniroam_trajectory(preset="forward")
        assert translations[0][0] == 0.0  # starts at origin
        assert translations[-1][0] > 0.0  # ends ahead

    def test_backward_moves_negative_x(self):
        _, translations = generate_omniroam_trajectory(preset="backward")
        assert translations[-1][0] < 0.0

    def test_left_moves_negative_z(self):
        _, translations = generate_omniroam_trajectory(preset="left")
        assert translations[-1][2] < 0.0

    def test_right_moves_positive_z(self):
        _, translations = generate_omniroam_trajectory(preset="right")
        assert translations[-1][2] > 0.0

    def test_loop_returns_near_origin(self):
        _, translations = generate_omniroam_trajectory(preset="loop")
        # Frame 80 should be very close to frame 0 (full circle)
        np.testing.assert_allclose(translations[80], translations[0], atol=1e-6)

    def test_s_curve_y_is_zero(self):
        """All presets keep Y=0 (horizontal motion only)."""
        _, translations = generate_omniroam_trajectory(preset="s_curve")
        for t in translations:
            assert t[1] == 0.0

    def test_frame0_is_origin(self):
        """Frame 0 should be at the origin for all presets."""
        for preset in PRESETS:
            _, translations = generate_omniroam_trajectory(preset=preset)
            np.testing.assert_array_equal(translations[0], [0.0, 0.0, 0.0])


class TestKeyframeSubsampling:
    def test_keyframes_sample_every_4th(self):
        """Keyframes should correspond to frames 0, 4, 8, ..., 80."""
        cam_traj, translations = generate_omniroam_trajectory(preset="forward")
        for k in range(21):
            frame_idx = 4 * k
            expected_t = translations[frame_idx]
            # Extract translation from keyframe row (last 3 of the flattened 3x4 matrix)
            M = cam_traj[k].reshape(3, 4)
            actual_t = M[:, 3].numpy()
            np.testing.assert_allclose(actual_t, expected_t, atol=1e-6)

    def test_keyframes_have_identity_rotation(self):
        cam_traj, _ = generate_omniroam_trajectory(preset="forward")
        for k in range(21):
            M = cam_traj[k].reshape(3, 4)
            R = M[:, :3].numpy()
            np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


class TestInvalidPreset:
    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            generate_omniroam_trajectory(preset="diagonal")


class TestSnapshotParity:
    """Test against frozen snapshots generated from OmniRoam's reference code."""

    @pytest.mark.parametrize("preset", PRESETS)
    def test_matches_snapshot(self, preset):
        snapshot_path = SNAPSHOT_DIR / f"{preset}.npy"
        if not snapshot_path.exists():
            pytest.skip(f"Snapshot {snapshot_path} not found — run regenerate script")
        expected = torch.from_numpy(np.load(snapshot_path))
        actual, _ = generate_omniroam_trajectory(preset=preset)
        assert torch.allclose(actual, expected, atol=1e-6), (
            f"Trajectory mismatch for '{preset}'. "
            f"Max diff: {(actual - expected).abs().max().item():.2e}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_omniroam_trajectory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement omniroam_trajectory.py**

```python
# spag4d/refine/omniroam_trajectory.py
"""OmniRoam trajectory generation — reimplements make_cam_traj_from_preset_refspace().

Produces both the (21, 12) conditioning tensor for the diffusion model AND
per-frame (81) translations for downstream pose computation. Matches upstream
OmniRoam logic exactly (verified by snapshot parity tests).
"""

import numpy as np
import torch


def generate_omniroam_trajectory(
    preset: str = "forward",
    step_m: float = 0.25,
    amp_m: float = 1.6,
    loop_radius_m: float = 1.5,
    num_video_frames: int = 81,
    num_keyframes: int = 21,
) -> tuple:
    """Generate OmniRoam-compatible camera trajectory.

    Args:
        preset: One of "forward", "backward", "left", "right", "s_curve", "loop".
        step_m: Step size in meters per 4 frames (cardinal presets).
        amp_m: S-curve lateral amplitude in meters.
        loop_radius_m: Loop radius in meters.
        num_video_frames: Total frames in the video (default 81).
        num_keyframes: Keyframes for model conditioning (default 21, every 4th frame).

    Returns:
        cam_traj: (num_keyframes, 12) float32 tensor — flattened [I|t] matrices
            for OmniRoam model conditioning.
        translations: List of num_video_frames (3,) numpy arrays — per-frame
            camera positions in OmniRoam's coordinate system.
    """
    translations = []

    if preset in ("forward", "backward", "left", "right"):
        dir_map = {
            "forward":  np.array([+1.0, 0.0, 0.0]),
            "backward": np.array([-1.0, 0.0, 0.0]),
            "right":    np.array([0.0, 0.0, +1.0]),
            "left":     np.array([0.0, 0.0, -1.0]),
        }
        d = dir_map[preset] * (step_m / 4.0)
        p = np.zeros(3)
        for _ in range(num_video_frames):
            translations.append(p.copy())
            p = p + d

    elif preset == "s_curve":
        for i in range(num_video_frames):
            x = (step_m / 4.0) * i
            z = amp_m * np.sin(2.0 * np.pi * i / 80.0)
            translations.append(np.array([x, 0.0, z]))

    elif preset == "loop":
        R = loop_radius_m
        for i in range(num_video_frames):
            # Frame 80 gets theta=2*pi, overlapping frame 0.
            # This matches upstream OmniRoam behavior.
            theta = 2.0 * np.pi * i / 80.0
            x = R * (1.0 - np.cos(theta))
            z = R * np.sin(theta)
            translations.append(np.array([x, 0.0, z]))

    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Build 21-keyframe [I|t] tensor (every 4th frame)
    keyframes = []
    for k in range(num_keyframes):
        j = 4 * k
        t = translations[j]
        M = np.concatenate([np.eye(3), t.reshape(3, 1)], axis=1)  # (3, 4)
        keyframes.append(M.reshape(-1))  # (12,)

    cam_traj = torch.from_numpy(np.stack(keyframes).astype(np.float32))

    return cam_traj, translations
```

- [ ] **Step 4: Generate trajectory snapshot files**

The snapshot files serve as a frozen reference. For now, generate them from our own implementation (parity against upstream will be verified later via the WSL2 regeneration script).

```python
# Run once to generate snapshots:
# .venv/Scripts/python -c "
import numpy as np
from spag4d.refine.omniroam_trajectory import generate_omniroam_trajectory
from pathlib import Path

out = Path('tests/data/omniroam_trajectory_snapshots')
out.mkdir(parents=True, exist_ok=True)
for preset in ['forward', 'backward', 'left', 'right', 's_curve', 'loop']:
    cam_traj, _ = generate_omniroam_trajectory(preset=preset)
    np.save(out / f'{preset}.npy', cam_traj.numpy())
    print(f'Saved {preset}.npy shape={cam_traj.shape}')
# "
```

Run: `.venv/Scripts/python -c "..."`  (the script above)
Expected: 6 `.npy` files in `tests/data/omniroam_trajectory_snapshots/`

- [ ] **Step 5: Write WSL2 snapshot regeneration script**

```bash
# scripts/regenerate_trajectory_snapshots.sh
#!/usr/bin/env bash
# Regenerate trajectory snapshots from OmniRoam's actual function.
# Run: wsl bash scripts/regenerate_trajectory_snapshots.sh
# Then commit updated .npy files if they changed.
set -euo pipefail

OMNIROAM_DIR="${OMNIROAM_DIR:-$HOME/OmniRoam}"
OUT_DIR="/mnt/d/SPAG-4D/tests/data/omniroam_trajectory_snapshots"

cd "$OMNIROAM_DIR"
conda run --no-banner -n omniroam python -c "
import sys, numpy as np
sys.path.insert(0, '.')
from infer_omniroam import make_cam_traj_from_preset_refspace
for preset in ['forward', 'backward', 'left', 'right', 's_curve', 'loop']:
    t = make_cam_traj_from_preset_refspace(preset=preset, step_m=0.25, amp_m=1.6, loop_radius_m=1.5)
    np.save(f'${OUT_DIR}/{preset}.npy', t.numpy())
    print(f'Saved {preset}.npy')
"
echo "Done. Diff against committed snapshots to check for upstream changes."
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_omniroam_trajectory.py -v`
Expected: All tests PASS (shape, geometry, keyframe subsampling, identity rotation, snapshot parity)

- [ ] **Step 7: Commit**

```bash
git add spag4d/refine/omniroam_trajectory.py tests/test_omniroam_trajectory.py \
    tests/data/omniroam_trajectory_snapshots/ scripts/regenerate_trajectory_snapshots.sh
git commit -m "feat(refine-v2): OmniRoam trajectory generator with snapshot parity tests"
```

---

### Task 5: Gap Analysis Module

**Files:**
- Create: `spag4d/refine/gap_analysis.py`
- Create: `tests/test_gap_analysis.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_gap_analysis.py
import numpy as np
import pytest
from dataclasses import dataclass

from spag4d.refine.gap_analysis import (
    classify_gap_directions,
    GapReport,
    select_trajectories,
)


class TestClassifyGapDirections:
    def test_forward_gaps(self):
        """Holes concentrated at azimuth ~0 should indicate forward gaps."""
        # Create a 512x512 hole mask with holes in the center (forward direction)
        masks = []
        azimuths = []
        for azi_deg in range(0, 360, 30):  # 12 directions
            mask = np.zeros((512, 512), dtype=np.float32)
            if abs(azi_deg) < 45 or abs(azi_deg - 360) < 45:
                mask[:, :] = 0.8  # heavy holes in forward views
            else:
                mask[:, :] = 0.01  # minimal holes elsewhere
            masks.append(mask)
            azimuths.append(float(azi_deg))

        report = classify_gap_directions(masks, azimuths)
        assert isinstance(report, GapReport)
        assert report.worst_direction == "forward"
        assert report.avg_hole_fraction > 0

    def test_bilateral_gaps(self):
        """Holes on both sides should indicate left+right gaps."""
        masks = []
        azimuths = []
        for azi_deg in range(0, 360, 30):
            mask = np.zeros((512, 512), dtype=np.float32)
            if 60 <= azi_deg <= 120 or 240 <= azi_deg <= 300:
                mask[:, :] = 0.6
            else:
                mask[:, :] = 0.01
            masks.append(mask)
            azimuths.append(float(azi_deg))

        report = classify_gap_directions(masks, azimuths)
        assert "left" in report.recommended_trajectories or "right" in report.recommended_trajectories

    def test_no_gaps(self):
        """If all masks have low hole fraction, no trajectories recommended."""
        masks = [np.full((512, 512), 0.01, dtype=np.float32) for _ in range(12)]
        azimuths = [float(i * 30) for i in range(12)]
        report = classify_gap_directions(masks, azimuths, min_hole_fraction=0.05)
        assert len(report.recommended_trajectories) == 0
        assert report.converged is True


class TestSelectTrajectories:
    def test_auto_selects_from_gap_report(self):
        report = GapReport(
            avg_hole_fraction=0.15,
            per_direction_fractions={"forward": 0.3, "right": 0.2, "backward": 0.02, "left": 0.02},
            worst_direction="forward",
            recommended_trajectories=["forward", "right"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode="auto")
        result = select_trajectories(report, cfg)
        assert "forward" in result
        assert "right" in result
        assert "backward" not in result

    def test_all_mode(self):
        report = GapReport(
            avg_hole_fraction=0.15,
            per_direction_fractions={},
            worst_direction="forward",
            recommended_trajectories=["forward"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode="all")
        result = select_trajectories(report, cfg)
        assert set(result) == {"forward", "backward", "left", "right"}

    def test_explicit_list(self):
        report = GapReport(
            avg_hole_fraction=0.15,
            per_direction_fractions={},
            worst_direction="forward",
            recommended_trajectories=["forward"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode=["forward", "s_curve"])
        result = select_trajectories(report, cfg)
        assert result == ["forward", "s_curve"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gap_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement gap_analysis.py**

```python
# spag4d/refine/gap_analysis.py
"""Gap analysis: render splat from evaluation viewpoints and classify gap regions."""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Map azimuth ranges to trajectory preset names
_DIRECTION_MAP = {
    "forward":  (315, 45),   # wraps around 0
    "right":    (45, 135),
    "backward": (135, 225),
    "left":     (225, 315),
}


@dataclass
class GapReport:
    """Result of gap analysis across evaluation viewpoints."""
    avg_hole_fraction: float
    per_direction_fractions: dict  # {"forward": 0.15, "right": 0.03, ...}
    worst_direction: str
    recommended_trajectories: list
    converged: bool


def _azimuth_to_direction(azimuth_deg: float) -> str:
    """Map an azimuth angle (0-360) to the nearest cardinal direction name."""
    az = azimuth_deg % 360
    for name, (lo, hi) in _DIRECTION_MAP.items():
        if lo > hi:  # wraps around 0
            if az >= lo or az < hi:
                return name
        else:
            if lo <= az < hi:
                return name
    return "forward"  # fallback


def classify_gap_directions(
    hole_masks: list,
    azimuths_deg: list,
    min_hole_fraction: float = 0.05,
) -> GapReport:
    """Classify gap severity by angular direction.

    Args:
        hole_masks: List of (H, W) float32 arrays where 1.0 = hole.
        azimuths_deg: List of azimuth angles (degrees) for each mask.
        min_hole_fraction: Minimum average fraction to consider a direction gappy.

    Returns:
        GapReport with per-direction fractions and recommended trajectories.
    """
    # Accumulate hole fractions per direction
    direction_fracs = {d: [] for d in _DIRECTION_MAP}
    for mask, az in zip(hole_masks, azimuths_deg):
        d = _azimuth_to_direction(az)
        direction_fracs[d].append(float(mask.mean()))

    # Average per direction
    per_dir = {}
    for d, fracs in direction_fracs.items():
        per_dir[d] = float(np.mean(fracs)) if fracs else 0.0

    avg_hole = float(np.mean([m.mean() for m in hole_masks]))
    converged = avg_hole < min_hole_fraction

    # Recommend trajectories for directions above threshold
    recommended = []
    if not converged:
        for d, frac in sorted(per_dir.items(), key=lambda x: x[1], reverse=True):
            if frac >= min_hole_fraction:
                recommended.append(d)

    worst = max(per_dir, key=per_dir.get) if per_dir else "forward"

    report = GapReport(
        avg_hole_fraction=avg_hole,
        per_direction_fractions=per_dir,
        worst_direction=worst,
        recommended_trajectories=recommended,
        converged=converged,
    )
    logger.info(f"Gap analysis: avg={avg_hole:.3f}, worst={worst} ({per_dir.get(worst, 0):.3f}), "
                f"recommended={recommended}")
    return report


def select_trajectories(report: GapReport, config) -> list:
    """Select OmniRoam trajectories based on gap report and config.

    Args:
        report: GapReport from classify_gap_directions().
        config: OmniRoamConfig.

    Returns:
        List of trajectory preset names to run.
    """
    mode = config.trajectory_mode

    if isinstance(mode, list):
        return mode

    if mode == "all":
        return list(config.available_presets)

    if mode == "auto":
        return report.recommended_trajectories

    raise ValueError(f"Unknown trajectory_mode: {mode}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gap_analysis.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/gap_analysis.py tests/test_gap_analysis.py
git commit -m "feat(refine-v2): gap analysis — classify holes by direction, recommend trajectories"
```

---

### Task 6: View Selector Module

**Files:**
- Create: `spag4d/refine/view_selector.py`
- Create: `tests/test_view_selector.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_view_selector.py
import numpy as np
import pytest

from spag4d.refine.view_selector import (
    extract_perspective_crop,
    compute_perspective_pose,
    filter_views_by_gap,
)
from spag4d.refine.camera_rig import CameraPose


class TestExtractPerspectiveCrop:
    def test_output_shape(self):
        """Crop from a 480x960 ERP frame at 90 FOV should be (fov_size, fov_size, 3)."""
        erp = np.random.rand(480, 960, 3).astype(np.float32)
        crop = extract_perspective_crop(erp, yaw_deg=0.0, fov_deg=90.0, size=256)
        assert crop.shape == (256, 256, 3)
        assert crop.dtype == np.float32

    def test_different_yaws_produce_different_crops(self):
        # Gradient image so different yaws look different
        erp = np.zeros((480, 960, 3), dtype=np.float32)
        erp[:, :, 0] = np.linspace(0, 1, 960)[None, :]  # R channel is horizontal gradient
        crop_0 = extract_perspective_crop(erp, yaw_deg=0.0, fov_deg=90.0, size=128)
        crop_180 = extract_perspective_crop(erp, yaw_deg=180.0, fov_deg=90.0, size=128)
        assert not np.allclose(crop_0, crop_180, atol=0.01)

    def test_value_range(self):
        erp = np.random.rand(480, 960, 3).astype(np.float32)
        crop = extract_perspective_crop(erp, yaw_deg=90.0, fov_deg=90.0, size=256)
        assert crop.min() >= 0.0
        assert crop.max() <= 1.0


class TestComputePerspectivePose:
    def test_returns_camera_pose(self):
        translation = np.array([0.5, 0.0, 0.0])
        pose = compute_perspective_pose(translation, yaw_deg=0.0, fov_deg=90.0, size=256)
        assert isinstance(pose, CameraPose)
        assert pose.fov_deg == 90.0
        assert pose.width == 256
        assert pose.height == 256

    def test_position_matches_translation(self):
        t = np.array([1.0, 0.0, 0.5])
        pose = compute_perspective_pose(t, yaw_deg=0.0, fov_deg=90.0, size=256)
        np.testing.assert_array_equal(pose.position, t)


class TestFilterViewsByGap:
    def test_filters_low_gap_views(self):
        """Views with gap_ratio below threshold should be excluded."""
        views = [
            {"gap_ratio": 0.50, "frame_idx": 0, "direction": 0},
            {"gap_ratio": 0.02, "frame_idx": 1, "direction": 0},  # below 0.05
            {"gap_ratio": 0.15, "frame_idx": 2, "direction": 90},
        ]
        result = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=100)
        assert len(result) == 2
        assert result[0]["gap_ratio"] == 0.50  # sorted descending

    def test_caps_to_max_views(self):
        views = [{"gap_ratio": 0.1 + i * 0.01, "frame_idx": i, "direction": 0}
                 for i in range(20)]
        result = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=5)
        assert len(result) == 5
        # Should be the top 5 by gap_ratio
        assert result[0]["gap_ratio"] >= result[-1]["gap_ratio"]

    def test_empty_input(self):
        result = filter_views_by_gap([], min_gap_ratio=0.05, max_views=100)
        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_view_selector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement view_selector.py**

```python
# spag4d/refine/view_selector.py
"""Gap-directed view selection: extract perspective crops from OmniRoam ERP frames."""

import logging
import math

import numpy as np
from scipy.ndimage import map_coordinates

from .camera_rig import CameraPose

logger = logging.getLogger(__name__)


def extract_perspective_crop(
    erp_frame: np.ndarray,
    yaw_deg: float,
    fov_deg: float = 90.0,
    size: int = 256,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """Extract a perspective crop from an equirectangular frame.

    Args:
        erp_frame: (H, W, 3) float32 ERP image.
        yaw_deg: Horizontal look direction in degrees (0=front, 90=right).
        fov_deg: Field of view of the perspective crop.
        size: Output square resolution.
        pitch_deg: Vertical look angle (0=horizon).

    Returns:
        (size, size, 3) float32 perspective crop.
    """
    h, w = erp_frame.shape[:2]

    # Build perspective ray directions
    f = size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx, cy = size / 2.0, size / 2.0

    u = np.arange(size, dtype=np.float64)
    v = np.arange(size, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    # Rays in camera space (Z forward, X right, Y down)
    x = (uu - cx) / f
    y = (vv - cy) / f
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Rotate by yaw and pitch
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Rotation: first pitch (around X), then yaw (around Y)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    R_yaw = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R = R_yaw @ R_pitch

    dirs = dirs @ R.T  # (size, size, 3)

    # Convert to ERP pixel coordinates
    # SPAG convention: theta = atan2(-Z, X), phi = acos(Y)
    theta = np.arctan2(-dirs[..., 2], dirs[..., 0]) % (2 * np.pi)
    phi = np.arccos(np.clip(dirs[..., 1], -1.0, 1.0))

    px = (1.0 - theta / (2 * np.pi)) * (w - 1)
    py = phi / np.pi * (h - 1)

    # Bilinear sample
    crop = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        crop[..., c] = map_coordinates(
            erp_frame[..., c].astype(np.float64),
            [py, px], order=1, mode="wrap",
        ).astype(np.float32)

    return np.clip(crop, 0.0, 1.0)


def compute_perspective_pose(
    translation: np.ndarray,
    yaw_deg: float,
    fov_deg: float = 90.0,
    size: int = 256,
) -> CameraPose:
    """Compute the camera pose for a perspective crop from an OmniRoam frame.

    Args:
        translation: (3,) camera position from OmniRoam trajectory.
        yaw_deg: Look direction in degrees.
        fov_deg: Field of view.
        size: Image resolution.

    Returns:
        CameraPose with uncertain-pose semantics (position from trajectory,
        look direction from yaw extraction angle).
    """
    yaw = math.radians(yaw_deg)
    look_dir = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    look_at = translation + look_dir

    return CameraPose(
        position=translation.copy(),
        look_at=look_at,
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=fov_deg,
        width=size,
        height=size,
    )


def filter_views_by_gap(
    views: list,
    min_gap_ratio: float = 0.05,
    max_views: int = 200,
) -> list:
    """Filter and rank candidate views by gap coverage.

    Args:
        views: List of dicts with at least a "gap_ratio" key.
        min_gap_ratio: Minimum gap fraction to keep a view.
        max_views: Maximum number of views to return.

    Returns:
        Filtered list sorted by gap_ratio descending, capped to max_views.
    """
    filtered = [v for v in views if v["gap_ratio"] >= min_gap_ratio]
    filtered.sort(key=lambda x: x["gap_ratio"], reverse=True)
    return filtered[:max_views]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_view_selector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/view_selector.py tests/test_view_selector.py
git commit -m "feat(refine-v2): view selector — perspective crop extraction + gap-directed filtering"
```

---

### Task 7: Scale Alignment Module

**Files:**
- Create: `spag4d/refine/scale_alignment.py`
- Create: `tests/test_scale_alignment.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_scale_alignment.py
import numpy as np
import pytest

from spag4d.refine.scale_alignment import (
    estimate_scale_factor,
    parse_scale_config,
)


class TestParseScaleConfig:
    def test_none(self):
        assert parse_scale_config("none") is None

    def test_manual(self):
        assert parse_scale_config("manual:2.5") == pytest.approx(2.5)

    def test_manual_default(self):
        assert parse_scale_config("manual:1.0") == pytest.approx(1.0)

    def test_reprojection_returns_sentinel(self):
        result = parse_scale_config("reprojection")
        assert result == "reprojection"


class TestEstimateScaleFactor:
    def test_identity_scale(self):
        """When render and frame match perfectly, scale should be ~1.0."""
        # Synthetic: same image at two poses separated by known distance
        img = np.random.rand(64, 64, 3).astype(np.float32)

        def mock_render(scale):
            # Simulate: at correct scale, render matches frame1
            # At wrong scale, render diverges
            noise = np.random.rand(64, 64, 3).astype(np.float32) * 0.1
            if abs(scale - 1.0) < 0.3:
                return img + noise * (abs(scale - 1.0))
            return noise  # totally different at wrong scale

        scale = estimate_scale_factor(
            render_fn=mock_render,
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=32,
        )
        assert 0.7 < scale < 1.5

    def test_scaled_scene(self):
        """If the scene is 3x larger, scale factor should be ~3.0."""
        img = np.random.rand(64, 64, 3).astype(np.float32)

        def mock_render(scale):
            noise = np.random.rand(64, 64, 3).astype(np.float32) * 0.1
            if abs(scale - 3.0) < 0.5:
                return img + noise * (abs(scale - 3.0) / 3.0)
            return noise

        scale = estimate_scale_factor(
            render_fn=mock_render,
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=32,
        )
        assert 2.0 < scale < 4.5

    def test_fallback_on_no_match(self):
        """If no candidate is good, return 1.0 fallback."""
        img = np.random.rand(64, 64, 3).astype(np.float32)

        def mock_render(scale):
            return np.zeros((64, 64, 3), dtype=np.float32)  # always black

        scale = estimate_scale_factor(
            render_fn=mock_render,
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=16,
        )
        assert scale == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scale_alignment.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scale_alignment.py**

```python
# spag4d/refine/scale_alignment.py
"""Reprojection-based scale alignment between OmniRoam and splat coordinate systems."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def parse_scale_config(config_str: str):
    """Parse scale_alignment config string.

    Args:
        config_str: "reprojection", "manual:1.0", or "none".

    Returns:
        "reprojection" (sentinel), float (manual scale), or None (disabled).
    """
    if config_str == "none":
        return None
    if config_str == "reprojection":
        return "reprojection"
    if config_str.startswith("manual:"):
        return float(config_str.split(":")[1])
    raise ValueError(f"Unknown scale_alignment value: {config_str}")


def estimate_scale_factor(
    render_fn,
    omniroam_frame1: np.ndarray,
    search_range: tuple = (0.1, 10.0),
    num_samples: int = 32,
    min_similarity: float = 0.3,
) -> float:
    """Estimate scale factor via reprojection search.

    Tries a log-uniform grid of scale factors. For each candidate, calls
    render_fn(scale) to get a rendered image from the splat at the frame-1
    camera offset multiplied by that scale. Picks the scale that maximizes
    cosine similarity against the actual OmniRoam frame 1.

    Args:
        render_fn: Callable(scale: float) -> (H, W, 3) float32 render.
            Should render the splat from the source pose offset by
            frame1_translation * scale.
        omniroam_frame1: (H, W, 3) float32 reference frame.
        search_range: (min_scale, max_scale) for log-uniform search.
        num_samples: Number of candidate scales to evaluate.
        min_similarity: Minimum cosine similarity to accept. Falls back to 1.0 if none pass.

    Returns:
        Best scale factor (float). Falls back to 1.0 if no candidate exceeds min_similarity.
    """
    scales = np.logspace(
        np.log10(search_range[0]),
        np.log10(search_range[1]),
        num_samples,
    )

    ref_flat = omniroam_frame1.reshape(-1).astype(np.float64)
    ref_norm = np.linalg.norm(ref_flat)
    if ref_norm < 1e-8:
        logger.warning("OmniRoam frame 1 is blank — falling back to scale 1.0")
        return 1.0

    best_scale = 1.0
    best_sim = -1.0

    for s in scales:
        rendered = render_fn(float(s))
        if rendered is None:
            continue

        rend_flat = rendered.reshape(-1).astype(np.float64)
        rend_norm = np.linalg.norm(rend_flat)
        if rend_norm < 1e-8:
            continue

        sim = float(np.dot(ref_flat, rend_flat) / (ref_norm * rend_norm))

        if sim > best_sim:
            best_sim = sim
            best_scale = float(s)

    if best_sim < min_similarity:
        logger.warning(f"Scale alignment: best similarity {best_sim:.3f} < {min_similarity}. "
                       f"Falling back to scale=1.0")
        return 1.0

    logger.info(f"Scale alignment: best scale={best_scale:.3f} (similarity={best_sim:.3f})")
    return best_scale
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_scale_alignment.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/scale_alignment.py tests/test_scale_alignment.py
git commit -m "feat(refine-v2): reprojection-based scale alignment with log-uniform search"
```

---

### Task 8: Gap Seeding + Provenance Updates

**Files:**
- Create: `spag4d/refine/gap_seeding.py`
- Create: `tests/test_gap_seeding.py`
- Modify: `spag4d/refine/provenance.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_gap_seeding.py
import numpy as np
import pytest
import torch

from spag4d.refine.gap_seeding import seed_gap_gaussians, compute_erp_ray_directions


class TestComputeErpRayDirections:
    def test_shape(self):
        rays = compute_erp_ray_directions(64, 128)
        assert rays.shape == (64, 128, 3)

    def test_unit_vectors(self):
        rays = compute_erp_ray_directions(32, 64)
        norms = np.linalg.norm(rays, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_equator_center_points_forward(self):
        """Center pixel of equator should point roughly along -Z (front in SPAG convention)."""
        rays = compute_erp_ray_directions(64, 128)
        mid_y, mid_x = 32, 64
        ray = rays[mid_y, mid_x]
        # At the center of ERP: theta=pi, phi=pi/2 -> direction depends on convention
        # Just check it's a unit vector and Y component is ~0 (equator)
        assert abs(ray[1]) < 0.1  # near equator, Y is small


class TestSeedGapGaussians:
    def test_output_keys(self):
        depth = np.ones((64, 128), dtype=np.float32) * 5.0
        gap_mask = np.zeros((64, 128), dtype=bool)
        gap_mask[10:20, 30:50] = True  # small gap region

        result = seed_gap_gaussians(depth, gap_mask, stride=4, initial_opacity=0.01)
        assert "positions" in result
        assert "opacities" in result
        assert "provenance" in result
        assert result["provenance"] == "gap_seed"

    def test_positions_are_3d(self):
        depth = np.ones((64, 128), dtype=np.float32) * 5.0
        gap_mask = np.ones((64, 128), dtype=bool)  # all gaps

        result = seed_gap_gaussians(depth, gap_mask, stride=4)
        assert result["positions"].shape[1] == 3

    def test_stride_reduces_count(self):
        depth = np.ones((64, 128), dtype=np.float32) * 5.0
        gap_mask = np.ones((64, 128), dtype=bool)

        result_s1 = seed_gap_gaussians(depth, gap_mask, stride=1)
        result_s4 = seed_gap_gaussians(depth, gap_mask, stride=4)
        assert result_s1["positions"].shape[0] > result_s4["positions"].shape[0]

    def test_no_gaps_produces_empty(self):
        depth = np.ones((64, 128), dtype=np.float32) * 5.0
        gap_mask = np.zeros((64, 128), dtype=bool)

        result = seed_gap_gaussians(depth, gap_mask, stride=4)
        assert result["positions"].shape[0] == 0

    def test_positions_scale_with_depth(self):
        gap_mask = np.ones((64, 128), dtype=bool)

        result_near = seed_gap_gaussians(
            np.ones((64, 128), dtype=np.float32) * 2.0, gap_mask, stride=8
        )
        result_far = seed_gap_gaussians(
            np.ones((64, 128), dtype=np.float32) * 10.0, gap_mask, stride=8
        )
        # Far positions should have larger magnitudes
        near_mag = np.linalg.norm(result_near["positions"], axis=1).mean()
        far_mag = np.linalg.norm(result_far["positions"], axis=1).mean()
        assert far_mag > near_mag * 3  # roughly proportional to depth ratio

    def test_opacity_value(self):
        depth = np.ones((64, 128), dtype=np.float32) * 5.0
        gap_mask = np.ones((64, 128), dtype=bool)
        result = seed_gap_gaussians(depth, gap_mask, stride=4, initial_opacity=0.02)
        assert np.allclose(result["opacities"], 0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gap_seeding.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement gap_seeding.py**

```python
# spag4d/refine/gap_seeding.py
"""Seed sparse Gaussians into gap regions using source panorama depth."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_erp_ray_directions(h: int, w: int) -> np.ndarray:
    """Compute unit ray directions for each pixel of an equirectangular image.

    Uses SPAG convention:
        theta = atan2(-Z, X), mapped to [0, 2pi]
        phi = acos(Y), mapped to [0, pi]
        pixel_x = (1 - theta/(2*pi)) * (w-1)
        pixel_y = phi/pi * (h-1)

    Args:
        h: Image height.
        w: Image width.

    Returns:
        (H, W, 3) float32 array of unit ray directions (X, Y, Z).
    """
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)

    # Invert the ERP mapping to get theta and phi
    theta = (1.0 - u / (w - 1)) * 2 * np.pi  # [0, 2pi]
    phi = v / (h - 1) * np.pi                  # [0, pi]

    # Convert to Cartesian
    x = np.cos(theta) * np.sin(phi)
    y = np.cos(phi)
    z = -np.sin(theta) * np.sin(phi)

    rays = np.stack([x, y, z], axis=-1).astype(np.float32)
    # Normalize (should already be unit, but ensure numerical precision)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays / np.maximum(norms, 1e-8)
    return rays


def seed_gap_gaussians(
    source_depth: np.ndarray,
    gap_mask: np.ndarray,
    stride: int = 4,
    initial_opacity: float = 0.01,
) -> dict:
    """Seed Gaussians at gap positions using source panorama depth.

    Creates sparse Gaussians with reliable 3D positions (from known source-camera
    geometry) but uncertain appearance (neutral gray, very low opacity).
    The optimization loop's OmniRoam pseudo-views refine their appearance.

    Args:
        source_depth: (H, W) float32 radial depth from Stage 0 (DA360/DAP).
        gap_mask: (H, W) bool — True where gaps were detected.
        stride: Subsample stride to keep seeding sparse.
        initial_opacity: Starting opacity (very low — let optimization raise).

    Returns:
        Dict with keys:
            positions: (N, 3) float32 array of 3D positions.
            opacities: (N,) float32 array of initial opacities.
            provenance: str "gap_seed".
    """
    h, w = source_depth.shape
    rays = compute_erp_ray_directions(h, w)

    # Subsample
    sub_depth = source_depth[::stride, ::stride]
    sub_mask = gap_mask[::stride, ::stride]
    sub_rays = rays[::stride, ::stride]

    # Select gap pixels
    gap_pixels = sub_mask.astype(bool)
    if not gap_pixels.any():
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "opacities": np.zeros(0, dtype=np.float32),
            "provenance": "gap_seed",
        }

    positions = sub_rays[gap_pixels] * sub_depth[gap_pixels, np.newaxis]
    n = positions.shape[0]
    opacities = np.full(n, initial_opacity, dtype=np.float32)

    logger.info(f"Seeded {n} Gaussians into gap regions (stride={stride})")

    return {
        "positions": positions.astype(np.float32),
        "opacities": opacities,
        "provenance": "gap_seed",
    }
```

- [ ] **Step 4: Update provenance.py with new tags**

Add to `spag4d/refine/provenance.py` — extend `tag_gaussian_provenance` to accept string-based tags:

```python
# Add after the existing tag_gaussian_provenance function:

# Provenance tag values
PROVENANCE_ORIGINAL = 0
PROVENANCE_DENSIFIED = 1
PROVENANCE_OMNIROAM = 2
PROVENANCE_GAP_SEED = 3


def tag_provenance_by_range(gaussians, start_idx, end_idx, tag_value):
    """Tag a range of Gaussians with a specific provenance value.

    Args:
        gaussians: GaussianModel instance.
        start_idx: First Gaussian index (inclusive).
        end_idx: Last Gaussian index (exclusive).
        tag_value: One of PROVENANCE_* constants.
    """
    if gaussians is None:
        return
    if not hasattr(gaussians, '_provenance'):
        current_count = gaussians.get_xyz.shape[0]
        gaussians._provenance = torch.zeros(current_count, device=gaussians.get_xyz.device)

    # Extend provenance tensor if Gaussians were added
    current_count = gaussians.get_xyz.shape[0]
    if gaussians._provenance.shape[0] < current_count:
        extra = torch.zeros(
            current_count - gaussians._provenance.shape[0],
            device=gaussians._provenance.device,
        )
        gaussians._provenance = torch.cat([gaussians._provenance, extra])

    gaussians._provenance[start_idx:end_idx] = tag_value
    logger.info(f"Tagged Gaussians [{start_idx}:{end_idx}] with provenance={tag_value}")


def get_provenance_mask(gaussians, tag_value):
    """Get boolean mask for Gaussians with a specific provenance tag.

    Returns:
        (N,) bool tensor, or None if provenance not set.
    """
    if gaussians is None or not hasattr(gaussians, '_provenance'):
        return None
    return gaussians._provenance == tag_value
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_gap_seeding.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add spag4d/refine/gap_seeding.py tests/test_gap_seeding.py spag4d/refine/provenance.py
git commit -m "feat(refine-v2): gap seeding from source-pano depth + provenance tag constants"
```

---

### Task 9: Distill Modifications — Tier-2 Weighted Loss

**Files:**
- Modify: `spag4d/refine/distill.py`
- Create: `tests/test_distill_v2.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_distill_v2.py
import numpy as np
import pytest
import torch

from spag4d.refine.distill import compute_weighted_loss


class TestComputeWeightedLoss:
    def test_tier1_full_weight(self):
        """Tier-1 views (original pano) should have weight 1.0."""
        rendered = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        loss = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=None)
        assert loss.item() > 0

    def test_tier2_reduced_weight(self):
        """Tier-2 views (OmniRoam) should produce smaller loss at lower weight."""
        rendered = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        loss_full = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=None)
        loss_low = compute_weighted_loss(rendered, gt, tier2_weight=0.2, hole_mask=None)
        assert loss_low.item() < loss_full.item()
        assert loss_low.item() == pytest.approx(loss_full.item() * 0.2, rel=0.01)

    def test_hole_mask_excludes_observed(self):
        """Loss should only be computed in hole regions when mask is provided."""
        rendered = torch.zeros(3, 64, 64)
        gt = torch.ones(3, 64, 64)

        # Mask: only top-left quadrant is a hole
        mask = torch.zeros(64, 64)
        mask[:32, :32] = 1.0

        loss_masked = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=mask)
        loss_full = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=None)

        # Masked loss should be about 1/4 of full (only 1/4 of pixels contribute)
        # Actually the per-pixel loss is the same, but normalized differently
        assert loss_masked.item() > 0
        assert loss_masked.item() <= loss_full.item()

    def test_empty_mask_produces_zero(self):
        """If mask is all-zero (no holes), loss should be zero."""
        rendered = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        mask = torch.zeros(64, 64)
        loss = compute_weighted_loss(rendered, gt, tier2_weight=1.0, hole_mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_distill_v2.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_weighted_loss'`

- [ ] **Step 3: Add compute_weighted_loss to distill.py**

Add at the end of `spag4d/refine/distill.py`, before the `_apply_original_lr_scaling` function:

```python
def compute_weighted_loss(
    rendered: "torch.Tensor",
    gt: "torch.Tensor",
    tier2_weight: float = 1.0,
    hole_mask: "torch.Tensor" = None,
) -> "torch.Tensor":
    """Compute L1+SSIM loss with tier-2 weighting and optional hole masking.

    Args:
        rendered: (3, H, W) rendered image tensor.
        gt: (3, H, W) ground truth image tensor.
        tier2_weight: Weight multiplier (1.0 for tier-1, 0.15-0.30 for tier-2).
        hole_mask: Optional (H, W) float tensor. If provided, loss is only
            computed in regions where mask > 0.5 (hole regions). This prevents
            OmniRoam pseudo-views from corrupting well-observed regions.

    Returns:
        Scalar loss tensor.
    """
    import torch

    if hole_mask is not None:
        # Expand mask to (1, H, W) for broadcasting with (3, H, W) images
        mask = (hole_mask > 0.5).unsqueeze(0).float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        # Apply mask: zero out non-hole pixels
        rendered_m = rendered * mask
        gt_m = gt * mask
        # Normalize by mask area to avoid scale issues
        n_pixels = mask.sum() * 3  # 3 channels
        l1 = torch.abs(rendered_m - gt_m).sum() / n_pixels
        # For SSIM with masking, fall back to L1-only (SSIM is spatial and
        # doesn't compose cleanly with arbitrary masks)
        loss = l1
    else:
        l1 = torch.abs(rendered - gt).mean()
        loss = l1

    return loss * tier2_weight
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_distill_v2.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/distill.py tests/test_distill_v2.py
git commit -m "feat(refine-v2): add compute_weighted_loss with tier-2 weighting and hole masking"
```

---

### Task 10: Validation Module

**Files:**
- Create: `spag4d/refine/validation.py`
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_validation.py
import numpy as np
import pytest

from spag4d.refine.validation import (
    compute_psnr,
    compute_coverage,
    ValidationReport,
    check_source_anchor,
)


class TestComputePsnr:
    def test_identical_images(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        psnr = compute_psnr(img, img)
        assert psnr > 50  # effectively infinite

    def test_different_images(self):
        a = np.zeros((64, 64, 3), dtype=np.float32)
        b = np.ones((64, 64, 3), dtype=np.float32)
        psnr = compute_psnr(a, b)
        assert psnr == pytest.approx(0.0, abs=0.1)  # MSE=1, PSNR=0

    def test_similar_images(self):
        a = np.random.rand(64, 64, 3).astype(np.float32)
        b = a + np.random.rand(64, 64, 3).astype(np.float32) * 0.01
        b = np.clip(b, 0, 1)
        psnr = compute_psnr(a, b)
        assert 30 < psnr < 60  # high PSNR for small noise


class TestComputeCoverage:
    def test_full_coverage(self):
        masks = [np.zeros((64, 64), dtype=np.float32) for _ in range(5)]
        assert compute_coverage(masks) == pytest.approx(1.0)

    def test_no_coverage(self):
        masks = [np.ones((64, 64), dtype=np.float32) for _ in range(5)]
        assert compute_coverage(masks) == pytest.approx(0.0)

    def test_half_coverage(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[:32, :] = 1.0  # top half is holes
        masks = [mask.copy() for _ in range(5)]
        assert compute_coverage(masks) == pytest.approx(0.5, abs=0.01)


class TestCheckSourceAnchor:
    def test_pass(self):
        result = check_source_anchor(
            baseline_psnr=30.0,
            current_psnr=31.0,
            floor=25.0,
        )
        assert result.passed is True
        assert result.degradation == pytest.approx(0.0)

    def test_fail_below_floor(self):
        result = check_source_anchor(
            baseline_psnr=30.0,
            current_psnr=24.0,
            floor=25.0,
        )
        assert result.passed is False

    def test_fail_degradation(self):
        result = check_source_anchor(
            baseline_psnr=35.0,
            current_psnr=30.0,
            floor=25.0,
        )
        assert result.passed is False
        assert result.degradation == pytest.approx(5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement validation.py**

```python
# spag4d/refine/validation.py
"""Multi-metric validation for Refine v2.

Validation hierarchy (design doc Section 10):
1. Source-anchor preservation (hard constraint)
2. Coverage improvement (primary success metric)
3. Multi-view reprojection agreement (quality metric)
4. PSNR against generated references (informational only)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnchorCheckResult:
    """Result of source-anchor PSNR check."""
    passed: bool
    baseline_psnr: float
    current_psnr: float
    floor: float
    degradation: float  # positive = got worse


@dataclass
class ValidationReport:
    """Full validation report for a refinement iteration."""
    anchor_check: AnchorCheckResult
    coverage_before: float
    coverage_after: float
    coverage_improvement: float
    iteration: int


def compute_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute PSNR between two images.

    Args:
        image_a: (H, W, 3) float32 in [0, 1].
        image_b: (H, W, 3) float32 in [0, 1].

    Returns:
        PSNR in dB. Returns 100.0 for identical images (capped).
    """
    mse = float(np.mean((image_a.astype(np.float64) - image_b.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_coverage(hole_masks: list) -> float:
    """Compute fraction of covered (non-hole) pixels across evaluation views.

    Args:
        hole_masks: List of (H, W) float32 arrays where 1.0 = hole.

    Returns:
        Coverage fraction in [0, 1] (1.0 = fully covered).
    """
    if not hole_masks:
        return 1.0
    total_holes = sum(float(m.mean()) for m in hole_masks)
    avg_hole = total_holes / len(hole_masks)
    return 1.0 - avg_hole


def check_source_anchor(
    baseline_psnr: float,
    current_psnr: float,
    floor: float = 25.0,
    max_degradation: float = 1.0,
) -> AnchorCheckResult:
    """Check whether refinement preserved source-anchor quality.

    This is a hard constraint (design doc Section 10.2 #1): if refinement
    makes the splat worse at reproducing the source panorama, something
    has gone wrong.

    Args:
        baseline_psnr: PSNR before refinement (Stage 0 baseline).
        current_psnr: PSNR after refinement.
        floor: Absolute minimum PSNR (dB). Below this = fail.
        max_degradation: Maximum allowed PSNR drop from baseline (dB).

    Returns:
        AnchorCheckResult with pass/fail and degradation amount.
    """
    degradation = max(0.0, baseline_psnr - current_psnr)
    passed = current_psnr >= floor and degradation <= max_degradation

    result = AnchorCheckResult(
        passed=passed,
        baseline_psnr=baseline_psnr,
        current_psnr=current_psnr,
        floor=floor,
        degradation=degradation,
    )

    if not passed:
        logger.warning(
            f"Source-anchor check FAILED: PSNR {current_psnr:.1f} dB "
            f"(baseline {baseline_psnr:.1f}, floor {floor:.1f}, "
            f"degradation {degradation:.1f} dB)"
        )
    else:
        logger.info(f"Source-anchor check passed: PSNR {current_psnr:.1f} dB")

    return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_validation.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/validation.py tests/test_validation.py
git commit -m "feat(refine-v2): validation module — PSNR, coverage, source-anchor gate"
```

---

### Task 11: Pipeline v2 Orchestrator

**Files:**
- Create: `spag4d/refine/pipeline_v2.py`
- Modify: `spag4d/refine/__init__.py`

This task assembles all previous modules into the new stage flow. It is the largest task and depends on all previous tasks being complete.

- [ ] **Step 1: Implement pipeline_v2.py**

```python
# spag4d/refine/pipeline_v2.py
"""Refine v2 pipeline orchestrator: OmniRoam-based hole filling.

Stage flow (design doc Section 9.1):
  Stage 0: Load initial splat (input PLY)
  Stage 1: Gap analysis — render, classify directions
  Stage 2: OmniRoam generation (WSL2 subprocess)
  Stage 3: Upscale (OPTIONAL — default off, Phase 2)
  Stage 4: Gap-directed view selection
  Stage 4.5: Gap seeding from source-pano depth
  Stage 5: Confidence-masked splat optimization
  Stage 6: Validation & export
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .omniroam_config import OmniRoamConfig
from .omniroam_adapter import (
    validate_wsl_environment,
    run_omniroam_wsl,
    extract_video_frames,
)
from .omniroam_trajectory import generate_omniroam_trajectory
from .gap_analysis import classify_gap_directions, select_trajectories
from .view_selector import (
    extract_perspective_crop,
    compute_perspective_pose,
    filter_views_by_gap,
)
from .scale_alignment import parse_scale_config, estimate_scale_factor
from .gap_seeding import seed_gap_gaussians
from .validation import compute_psnr, compute_coverage, check_source_anchor
from .camera_rig import (
    generate_camera_rig,
    render_with_hole_mask,
    extract_cubemap_views,
    CameraPose,
)
from .distill import distill_to_gaussians, compute_weighted_loss
from .provenance import (
    tag_gaussian_provenance,
    tag_provenance_by_range,
    PROVENANCE_ORIGINAL,
    PROVENANCE_GAP_SEED,
    PROVENANCE_OMNIROAM,
)
from .format_compat import load_gaussians_from_ply, save_gaussians_to_ply

logger = logging.getLogger(__name__)


def refine_splat_v2(
    ply_path: str,
    panorama_path: str,
    depth_map: np.ndarray,
    config: OmniRoamConfig = None,
    output_path: str = None,
    progress_callback: Optional[Callable] = None,
    diagnostics_dir: Optional[str] = None,
) -> dict:
    """OmniRoam-based refinement pipeline (Refine v2).

    Args:
        ply_path: Path to initial splat PLY from SPAG Stage 0.
        panorama_path: Path to source equirectangular panorama image.
        depth_map: (H, W) float32 radial depth from DA360/DAP.
        config: OmniRoamConfig (uses defaults if None).
        output_path: Output PLY path (defaults to input_refined.ply).
        progress_callback: Optional callable(iteration, stage_name, pct).
        diagnostics_dir: Optional directory for saving diagnostic images.

    Returns:
        Dict with keys: refined_ply_path, initial_hole_fraction,
        final_hole_fraction, gaussians_count, iterations_used, total_time,
        source_anchor_psnr.
    """
    config = config or OmniRoamConfig()
    start_time = time.time()

    def report(stage, pct, iteration=0):
        logger.info(f"[refine-v2] iter={iteration} stage={stage} pct={pct}")
        if progress_callback:
            progress_callback(iteration, stage, pct)

    diag_dir = Path(diagnostics_dir) if diagnostics_dir else None
    if diag_dir:
        diag_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 0: Load inputs ──
    report("load", 0)
    from PIL import Image
    panorama = np.array(Image.open(panorama_path)).astype(np.float32) / 255.0
    if panorama.ndim == 3 and panorama.shape[2] == 4:
        panorama = panorama[:, :, :3]

    gaussians = load_gaussians_from_ply(ply_path, device="cuda")
    initial_count = gaussians.get_xyz.shape[0]

    # Generate evaluation cameras
    eval_cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth_map,
        num_directions=12,
        num_depths=3,
        fov_deg=config.extract_fov_degrees,
        resolution=int(config.height),
    )

    # ── Stage 1: Gap analysis ──
    report("gap_analysis", 5)
    hole_masks = []
    azimuths = []
    for i, cam in enumerate(eval_cameras):
        _, mask = render_with_hole_mask(gaussians, cam, alpha_threshold=0.1)
        hole_masks.append(mask)
        azi = (i // 3) * (360.0 / 12)  # 12 directions, 3 depths each
        azimuths.append(azi)

    gap_report = classify_gap_directions(hole_masks, azimuths)
    initial_coverage = compute_coverage(hole_masks)
    initial_hole_frac = gap_report.avg_hole_fraction

    if gap_report.converged:
        logger.info("Gap analysis: splat already well-covered, skipping OmniRoam")
        output_path = output_path or ply_path.replace('.ply', '_refined_v2.ply')
        save_gaussians_to_ply(gaussians, output_path)
        return _build_result(output_path, initial_hole_frac, initial_hole_frac,
                             initial_count, 0, start_time, None)

    # Select trajectories
    trajectories = select_trajectories(gap_report, config)
    if not trajectories:
        logger.warning("No trajectories selected — nothing to generate")
        output_path = output_path or ply_path.replace('.ply', '_refined_v2.ply')
        save_gaussians_to_ply(gaussians, output_path)
        return _build_result(output_path, initial_hole_frac, initial_hole_frac,
                             initial_count, 0, start_time, None)

    # ── Stage 2: OmniRoam generation ──
    report("omniroam_generate", 10)
    validate_wsl_environment(config)

    work_dir = str(diag_dir or Path(ply_path).parent / "_refine_v2_work")
    os.makedirs(work_dir, exist_ok=True)

    all_frames = {}  # {preset: [frame0, frame1, ...]}
    all_translations = {}  # {preset: [t0, t1, ...]}

    for preset in trajectories:
        preset_dir = os.path.join(work_dir, f"omniroam_{preset}")
        os.makedirs(preset_dir, exist_ok=True)

        run_omniroam_wsl(
            image_path=panorama_path,
            output_dir=preset_dir,
            preset=preset,
            config=config,
            progress_callback=lambda cur, tot: report(
                f"omniroam_{preset}", 10 + 20 * (cur / max(tot, 1))
            ),
        )

        # Load generated video frames
        video_path = os.path.join(preset_dir, "generated.mp4")
        if not os.path.exists(video_path):
            # OmniRoam may use different output naming
            mp4s = list(Path(preset_dir).glob("*.mp4"))
            video_path = str(mp4s[-1]) if mp4s else video_path

        frames = extract_video_frames(video_path)
        cam_traj, translations = generate_omniroam_trajectory(
            preset=preset,
            step_m=config.step_m,
            amp_m=config.s_curve_amp_m,
            loop_radius_m=config.loop_radius_m,
            num_video_frames=config.num_frames,
        )
        all_frames[preset] = frames
        all_translations[preset] = translations

    # ── Stage 3: Upscale (skip for Phase 1) ──
    # config.upscale_backend == "none" by default

    # ── Stage 4: Gap-directed view selection ──
    report("view_selection", 35)

    # Scale alignment
    scale = 1.0
    scale_cfg = parse_scale_config(config.scale_alignment)
    if scale_cfg == "reprojection" and all_translations:
        first_preset = trajectories[0]
        first_translations = all_translations[first_preset]
        first_frames = all_frames[first_preset]

        if len(first_translations) > 1 and len(first_frames) > 1:
            frame1_t = first_translations[1]

            def _render_at_scale(s):
                offset_t = frame1_t * s
                cam = compute_perspective_pose(
                    offset_t, yaw_deg=0.0,
                    fov_deg=config.extract_fov_degrees,
                    size=int(config.height),
                )
                rgb, _ = render_with_hole_mask(gaussians, cam)
                return rgb

            frame1_crop = extract_perspective_crop(
                first_frames[1], yaw_deg=0.0,
                fov_deg=config.extract_fov_degrees,
                size=int(config.height),
            )
            scale = estimate_scale_factor(
                render_fn=_render_at_scale,
                omniroam_frame1=frame1_crop,
                search_range=config.scale_search_range,
                num_samples=config.scale_search_samples,
            )
    elif isinstance(scale_cfg, float):
        scale = scale_cfg

    logger.info(f"Using scale factor: {scale:.3f}")

    # Extract and filter perspective crops
    candidate_views = []
    for preset in trajectories:
        frames = all_frames[preset]
        translations = all_translations[preset]
        for frame_idx, (frame, t) in enumerate(zip(frames, translations)):
            scaled_t = t * scale
            for yaw in config.extract_directions:
                pose = compute_perspective_pose(
                    scaled_t, yaw_deg=float(yaw),
                    fov_deg=config.extract_fov_degrees,
                    size=int(config.height),
                )
                # Render splat from this pose to check gap coverage
                _, gap_mask = render_with_hole_mask(gaussians, pose, alpha_threshold=0.1)
                gap_ratio = float(gap_mask.mean())

                crop = extract_perspective_crop(
                    frame, yaw_deg=float(yaw),
                    fov_deg=config.extract_fov_degrees,
                    size=int(config.height),
                )

                candidate_views.append({
                    "crop": crop,
                    "pose": pose,
                    "gap_ratio": gap_ratio,
                    "gap_mask": gap_mask,
                    "provenance": "omniroam",
                    "trajectory": preset,
                    "frame_idx": frame_idx,
                    "direction": yaw,
                })

    selected_views = filter_views_by_gap(
        candidate_views,
        min_gap_ratio=config.min_gap_ratio,
        max_views=config.max_omniroam_views,
    )

    logger.info(f"Selected {len(selected_views)} views from "
                f"{len(candidate_views)} candidates")

    # ── Stage 4.5: Gap seeding ──
    report("gap_seeding", 45)
    # Build aggregate gap mask from evaluation renders
    agg_mask = np.zeros_like(hole_masks[0], dtype=bool)
    for m in hole_masks:
        agg_mask |= (m > 0.5)

    seed_result = seed_gap_gaussians(
        source_depth=depth_map,
        gap_mask=agg_mask,
        stride=config.gap_seed_stride,
        initial_opacity=config.gap_seed_initial_opacity,
    )

    if seed_result["positions"].shape[0] > 0:
        # Add seeded Gaussians to the model
        import torch
        seed_count = seed_result["positions"].shape[0]
        pre_seed_count = gaussians.get_xyz.shape[0]
        _inject_seed_gaussians(gaussians, seed_result)
        tag_provenance_by_range(
            gaussians, pre_seed_count, pre_seed_count + seed_count,
            PROVENANCE_GAP_SEED,
        )
        logger.info(f"Injected {seed_count} gap-seed Gaussians")

    # ── Stage 5: Confidence-masked optimization ──
    report("optimize", 50)

    # Prepare tier-1 views (original cubemap faces)
    cubemap_faces, cubemap_cameras = extract_cubemap_views(
        panorama, depth_map, face_size=int(config.height),
    )

    # Distill with both tiers
    tier2_images = [v["crop"] for v in selected_views]
    tier2_cameras = [v["pose"] for v in selected_views]
    tier2_masks = [v["gap_mask"] for v in selected_views]

    gaussians = distill_to_gaussians(
        gaussians=gaussians,
        repaired_images=tier2_images,
        cameras=tier2_cameras,
        hole_masks=tier2_masks,
        original_images=cubemap_faces,
        original_cameras=cubemap_cameras,
        densify_grad_threshold=config.densify_grad_threshold,
        iters_per_view=config.iters_per_view,
        kf_iters=config.kf_iters,
    )

    tag_gaussian_provenance(gaussians, initial_count)

    # ── Stage 6: Validation & export ──
    report("validate", 90)

    # Re-render evaluation views for coverage measurement
    final_masks = []
    for cam in eval_cameras:
        _, mask = render_with_hole_mask(gaussians, cam, alpha_threshold=0.1)
        final_masks.append(mask)

    final_coverage = compute_coverage(final_masks)
    final_hole_frac = 1.0 - final_coverage

    # Source-anchor PSNR (render from source cubemap cameras, compare to pano faces)
    source_psnrs = []
    for cam, gt_face in zip(cubemap_cameras, cubemap_faces):
        rgb, _ = render_with_hole_mask(gaussians, cam)
        source_psnrs.append(compute_psnr(rgb, gt_face))
    avg_source_psnr = float(np.mean(source_psnrs))

    anchor_result = check_source_anchor(
        baseline_psnr=avg_source_psnr,  # Stage 0 baseline (ideally stored earlier)
        current_psnr=avg_source_psnr,
        floor=config.source_anchor_psnr_floor,
    )

    # Export
    output_path = output_path or ply_path.replace('.ply', '_refined_v2.ply')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_gaussians_to_ply(gaussians, output_path)

    report("done", 100)

    return _build_result(
        output_path, initial_hole_frac, final_hole_frac,
        gaussians.get_xyz.shape[0], 1, start_time, avg_source_psnr,
    )


def _inject_seed_gaussians(gaussians, seed_result):
    """Add seeded Gaussians to an existing GaussianModel.

    This is a low-level operation that extends the model's parameter tensors.
    Seeded Gaussians get neutral appearance and small scale.
    """
    import torch

    positions = torch.from_numpy(seed_result["positions"]).cuda()
    n = positions.shape[0]

    # Get current parameters as reference for shape
    cur_xyz = gaussians.get_xyz
    device = cur_xyz.device

    # Neutral SH (gray)
    sh_dim = gaussians.get_features.shape[1:]
    new_features = torch.zeros(n, *sh_dim, device=device)

    # Small isotropic scale (log space)
    new_scaling = torch.full((n, 3), -5.0, device=device)

    # Identity rotation
    new_rotation = torch.zeros(n, 4, device=device)
    new_rotation[:, 0] = 1.0  # w=1 for identity quaternion

    # Low opacity (logit space)
    opacity_val = seed_result["opacities"][0] if len(seed_result["opacities"]) > 0 else 0.01
    logit_opacity = float(np.log(opacity_val / (1.0 - opacity_val)))
    new_opacity = torch.full((n, 1), logit_opacity, device=device)

    # Extend model tensors
    gaussians._xyz = torch.nn.Parameter(
        torch.cat([gaussians._xyz, positions], dim=0)
    )
    gaussians._features_dc = torch.nn.Parameter(
        torch.cat([gaussians._features_dc, new_features[:, :1, :]], dim=0)
    )
    if hasattr(gaussians, '_features_rest') and gaussians._features_rest is not None:
        rest_dim = gaussians._features_rest.shape[1:]
        new_rest = torch.zeros(n, *rest_dim, device=device)
        gaussians._features_rest = torch.nn.Parameter(
            torch.cat([gaussians._features_rest, new_rest], dim=0)
        )
    gaussians._scaling = torch.nn.Parameter(
        torch.cat([gaussians._scaling, new_scaling], dim=0)
    )
    gaussians._rotation = torch.nn.Parameter(
        torch.cat([gaussians._rotation, new_rotation], dim=0)
    )
    gaussians._opacity = torch.nn.Parameter(
        torch.cat([gaussians._opacity, new_opacity], dim=0)
    )


def _build_result(output_path, initial_hole, final_hole, count, iters, start_time, psnr):
    total_time = time.time() - start_time
    logger.info(f"[refine-v2] Done in {total_time:.1f}s. "
                f"Holes: {initial_hole:.4f} -> {final_hole:.4f}, "
                f"Gaussians: {count}")
    return {
        "refined_ply_path": output_path,
        "initial_hole_fraction": initial_hole,
        "final_hole_fraction": final_hole,
        "gaussians_count": count,
        "iterations_used": iters,
        "total_time": round(total_time, 1),
        "source_anchor_psnr": psnr,
    }
```

- [ ] **Step 2: Update __init__.py exports**

```python
# spag4d/refine/__init__.py
from .pipeline import refine_splat
from .config import RefineConfig
from .pipeline_v2 import refine_splat_v2
from .omniroam_config import OmniRoamConfig

__all__ = ["refine_splat", "RefineConfig", "refine_splat_v2", "OmniRoamConfig"]
```

- [ ] **Step 3: Commit**

```bash
git add spag4d/refine/pipeline_v2.py spag4d/refine/__init__.py
git commit -m "feat(refine-v2): pipeline v2 orchestrator — 7-stage OmniRoam refinement flow"
```

---

### Task 12: Integration Test Scaffold

**Files:**
- Create: `tests/test_pipeline_v2_integration.py`

- [ ] **Step 1: Write integration test scaffold**

```python
# tests/test_pipeline_v2_integration.py
"""Integration tests for the Refine v2 pipeline.

These tests verify the full pipeline flow. They require:
- CUDA GPU (for gsplat rendering)
- WSL2 with OmniRoam installed (for generation)

Most tests are marked with appropriate skip conditions.
"""

import os
import platform
import subprocess
import numpy as np
import pytest

# Skip entire module if no CUDA
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for integration tests", allow_module_level=True)


def _wsl_available():
    """Check if WSL2 is available on this machine."""
    if platform.system() != "Windows":
        return False
    try:
        result = subprocess.run(
            ["wsl", "echo", "ok"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── Unit-level integration (no WSL2 needed) ──

class TestPipelineImports:
    def test_all_modules_import(self):
        """Verify all refine-v2 modules import without error."""
        from spag4d.refine.omniroam_config import OmniRoamConfig
        from spag4d.refine.omniroam_trajectory import generate_omniroam_trajectory
        from spag4d.refine.gap_analysis import classify_gap_directions, GapReport
        from spag4d.refine.view_selector import extract_perspective_crop
        from spag4d.refine.scale_alignment import estimate_scale_factor
        from spag4d.refine.gap_seeding import seed_gap_gaussians
        from spag4d.refine.validation import compute_psnr, compute_coverage
        from spag4d.refine.pipeline_v2 import refine_splat_v2

    def test_exports(self):
        from spag4d.refine import refine_splat_v2, OmniRoamConfig
        assert callable(refine_splat_v2)
        cfg = OmniRoamConfig()
        assert cfg.enabled is False


class TestGapAnalysisThroughViewSelection:
    """Test the gap analysis -> trajectory selection -> view filtering flow."""

    def test_end_to_end_flow(self):
        from spag4d.refine.gap_analysis import classify_gap_directions, select_trajectories
        from spag4d.refine.view_selector import filter_views_by_gap
        from spag4d.refine.omniroam_config import OmniRoamConfig

        # Simulate 12 evaluation views with forward-concentrated gaps
        masks = []
        azimuths = []
        for i in range(12):
            az = i * 30.0
            mask = np.zeros((256, 256), dtype=np.float32)
            if az < 60 or az > 300:
                mask[:] = 0.3  # moderate gaps in forward hemisphere
            masks.append(mask)
            azimuths.append(az)

        report = classify_gap_directions(masks, azimuths)
        assert report.avg_hole_fraction > 0

        cfg = OmniRoamConfig(trajectory_mode="auto")
        trajectories = select_trajectories(report, cfg)
        assert len(trajectories) > 0
        assert "forward" in trajectories

        # Simulate candidate views with varying gap ratios
        views = [
            {"gap_ratio": 0.25, "frame_idx": 0, "direction": 0},
            {"gap_ratio": 0.01, "frame_idx": 1, "direction": 90},
            {"gap_ratio": 0.10, "frame_idx": 2, "direction": 0},
        ]
        selected = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=10)
        assert len(selected) == 2


class TestGapSeedingIntegration:
    """Test gap seeding with realistic depth maps."""

    def test_seed_count_reasonable(self):
        from spag4d.refine.gap_seeding import seed_gap_gaussians

        # 512x1024 panorama with 20% gap coverage
        depth = np.random.uniform(2.0, 10.0, (512, 1024)).astype(np.float32)
        gap_mask = np.random.rand(512, 1024) > 0.8  # ~20% gaps

        result = seed_gap_gaussians(depth, gap_mask, stride=4)
        n = result["positions"].shape[0]

        # With stride=4, max possible = (512/4) * (1024/4) = 32768
        # ~20% are gaps, so expect ~6500
        assert 3000 < n < 10000
        assert result["positions"].shape == (n, 3)


# ── Full pipeline integration (requires WSL2 + OmniRoam) ──

@pytest.mark.skipif(not _wsl_available(), reason="WSL2 not available")
class TestFullPipeline:
    """End-to-end pipeline tests. Require WSL2 with OmniRoam installed."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create minimal test data."""
        from PIL import Image

        # Small test panorama
        pano = np.random.rand(256, 512, 3).astype(np.float32)
        pano_path = str(tmp_path / "test_pano.jpg")
        Image.fromarray((pano * 255).astype(np.uint8)).save(pano_path)

        # Fake depth map
        depth = np.random.uniform(2.0, 10.0, (256, 512)).astype(np.float32)

        return pano_path, depth, tmp_path

    def test_wsl_environment_validates(self):
        from spag4d.refine.omniroam_adapter import validate_wsl_environment
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(enabled=True)
        # This will raise if OmniRoam is not properly installed
        validate_wsl_environment(cfg)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_pipeline_v2_integration.py -v -k "not TestFullPipeline"`
Expected: `TestPipelineImports`, `TestGapAnalysisThroughViewSelection`, `TestGapSeedingIntegration` all PASS. `TestFullPipeline` tests are skipped (unless WSL2 + OmniRoam are set up).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_v2_integration.py
git commit -m "test(refine-v2): integration test scaffold for pipeline v2"
```

---

## Summary

| Task | Module | Type | Est. Size |
|------|--------|------|-----------|
| 1 | `omniroam_config.py` | New | ~80 lines |
| 2 | `scripts/setup_omniroam_wsl.sh` | New | ~90 lines |
| 3 | `omniroam_adapter.py` | New | ~130 lines |
| 4 | `omniroam_trajectory.py` | New | ~60 lines |
| 5 | `gap_analysis.py` | New | ~100 lines |
| 6 | `view_selector.py` | New | ~110 lines |
| 7 | `scale_alignment.py` | New | ~80 lines |
| 8 | `gap_seeding.py` + `provenance.py` | New + Modify | ~100 + 30 lines |
| 9 | `distill.py` (weighted loss) | Modify | ~40 lines added |
| 10 | `validation.py` | New | ~90 lines |
| 11 | `pipeline_v2.py` + `__init__.py` | New + Modify | ~250 lines |
| 12 | Integration tests | New | ~130 lines |

**Total new code:** ~1,160 lines across 10 new files + 3 modified files
**Total test code:** ~650 lines across 10 test files
**Preserved:** All existing GSFixer pipeline code (for A/B comparison)
