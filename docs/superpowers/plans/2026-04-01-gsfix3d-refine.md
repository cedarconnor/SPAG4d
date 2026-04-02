# GSFix3D Refine Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-functional Klein-based refinement pipeline with GSFix3D-based disocclusion repair that uses render-repair-distill via differentiable rendering.

**Architecture:** New `spag4d/refine/` module with 3 phases: (1) camera rig generation + hole detection via novel-view rendering, (2) GSFixer diffusion model fine-tuning + inference for hole repair, (3) 3DGS distillation to optimize Gaussians against repaired images. GSFix3D vendored as git submodule in `third_party/GSFix3D/`.

**Tech Stack:** Python 3.10+, PyTorch 2.4, diff-gaussian-rasterization (CUDA), Stable Diffusion v2 (via GSFix3D's MarigoldGSFixerPipeline), trimesh, open3d, FastAPI, GaussianSplats3D viewer.

**Spec:** `docs/superpowers/specs/2026-04-01-gsfix3d-refine-design.md`

---

## File Map

### New files (create)

| File | Responsibility |
|------|---------------|
| `spag4d/refine/__init__.py` | Public API: `refine_splat()` |
| `spag4d/refine/pipeline.py` | Top-level orchestrator with iterative repair loop |
| `spag4d/refine/config.py` | `RefineConfig` dataclass with all hyperparameters |
| `spag4d/refine/camera_rig.py` | Camera placement, perspective rendering, hole mask extraction |
| `spag4d/refine/mesh_extract.py` | Depth-to-mesh via Poisson reconstruction for dual conditioning |
| `spag4d/refine/gsfixer_adapter.py` | GSFixer model load, scene fine-tune, inference wrapper |
| `spag4d/refine/distill.py` | 3DGS optimization loop (L1 + D-SSIM loss, densification) |
| `spag4d/refine/provenance.py` | Tag Gaussians as original vs. new, LR scaling |
| `spag4d/refine/format_compat.py` | PLY conversion between SPAG-4D and GSFix3D formats |
| `tests/test_refine_config.py` | Config defaults and validation |
| `tests/test_format_compat.py` | PLY round-trip consistency |
| `tests/test_camera_rig.py` | Camera geometry correctness |
| `tests/test_mesh_extract.py` | Mesh extraction from synthetic depth |

### Modified files

| File | Lines | What changes |
|------|-------|-------------|
| `api.py` | 398-582 | Rewire refine endpoints to new module, simplify parameters |
| `static/index.html` | 93-325 | Simplify refine panel, update help text, remove Klein controls |
| `static/js/app.js` | 384-505 | Simplify `startRefinement()`, update stage labels in `checkRefineStatus()` |
| `spag4d/cli.py` | 17-127 | Add `--refine` flag to convert command |
| `spag4d/cli.py` | 129-162 | Add GSFix3D checkpoint to `download-models` |

### Deleted files (Wave 5)

| Path | Reason |
|------|--------|
| `spag4d-refine/` (entire directory) | Replaced by `spag4d/refine/` |

---

## Wave 1: Scaffold (stubs + wiring)

### Task 1: Add GSFix3D submodule and build CUDA extension

**Files:**
- Create: `third_party/` (directory)
- Modify: `.gitmodules`

- [ ] **Step 1: Add GSFix3D as git submodule**

```bash
cd D:/SPAG-4D
git submodule add https://github.com/GSFix3D/GSFix3D.git third_party/GSFix3D
git submodule update --init --recursive
```

Expected: `third_party/GSFix3D/` directory with full repo including `diff-gaussian-rasterization/` submodule.

- [ ] **Step 2: Install GSFix3D Python dependencies**

```bash
cd D:/SPAG-4D
.venv/Scripts/pip install trimesh open3d plyfile omegaconf
```

Expected: Successful installation. Note: We skip their full requirements.txt since we already have torch, diffusers, etc. in .venv. We install only what's missing.

- [ ] **Step 3: Build diff-gaussian-rasterization**

```bash
cd D:/SPAG-4D/third_party/GSFix3D/diff-gaussian-rasterization
D:/SPAG-4D/.venv/Scripts/pip install .
```

Expected: CUDA extension compiles and installs. Verify:

```bash
D:/SPAG-4D/.venv/Scripts/python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('OK')"
```

- [ ] **Step 4: Commit submodule addition**

```bash
cd D:/SPAG-4D
git add .gitmodules third_party/GSFix3D
git commit -m "chore: add GSFix3D as submodule for disocclusion repair"
```

---

### Task 2: Create RefineConfig dataclass

**Files:**
- Create: `spag4d/refine/__init__.py`
- Create: `spag4d/refine/config.py`
- Test: `tests/test_refine_config.py`

- [ ] **Step 1: Write test for config defaults**

```python
# tests/test_refine_config.py
from spag4d.refine.config import RefineConfig


def test_config_defaults():
    cfg = RefineConfig()
    assert cfg.camera_fov == 60.0
    assert cfg.max_iterations == 3
    assert cfg.convergence_threshold == 0.02
    assert cfg.finetune_steps == 500
    assert cfg.distill_iterations == 3000
    assert len(cfg.translation_fracs) == 3


def test_config_override():
    cfg = RefineConfig(max_iterations=5, finetune_steps=200)
    assert cfg.max_iterations == 5
    assert cfg.finetune_steps == 200
    # Other defaults unchanged
    assert cfg.camera_fov == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/SPAG-4D
.venv/Scripts/python -m pytest tests/test_refine_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'spag4d.refine'`

- [ ] **Step 3: Create the module and config**

```python
# spag4d/refine/__init__.py
"""GSFix3D-based disocclusion repair for panorama-sourced 3DGS."""
```

```python
# spag4d/refine/config.py
"""Refinement pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RefineConfig:
    """All hyperparameters for the GSFix3D refinement pipeline."""

    # --- Paths ---
    gsfixer_checkpoint: str = "pretrained/gsfix3d/gsfix3d_base.ckpt"

    # --- Phase 1: Camera Rig ---
    camera_fov: float = 60.0
    translation_fracs: tuple = (0.05, 0.15, 0.30)
    alpha_threshold: float = 0.1
    num_directions: int = 12
    render_resolution: int = 512

    # --- Phase 1: Camera Selection ---
    min_hole_fraction: float = 0.03
    max_repair_cameras: int = 20

    # --- Phase 2: GSFixer ---
    finetune_steps: int = 500
    finetune_lr: float = 1.0e-5
    inference_steps: int = 50
    guidance_scale: float = 7.5
    mesh_simplify_ratio: float = 0.1

    # --- Phase 3: Distillation ---
    distill_iterations: int = 3000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    lr_position: float = 0.00016
    lr_feature: float = 0.0025
    lr_opacity: float = 0.05
    lr_scaling: float = 0.005
    lr_rotation: float = 0.001
    original_view_ratio: float = 0.3

    # --- Convergence ---
    convergence_threshold: float = 0.02
    max_iterations: int = 3

    # --- Progress callback stages ---
    STAGES: dict = field(default_factory=lambda: {
        "camera_rig": "Generating cameras",
        "mesh_extract": "Extracting mesh",
        "finetune": "Adapting to scene",
        "render_holes": "Detecting holes",
        "gsfixer_inference": "Repairing holes",
        "distill": "Optimizing 3D",
    })
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/Scripts/python -m pytest tests/test_refine_config.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/__init__.py spag4d/refine/config.py tests/test_refine_config.py
git commit -m "feat(refine): add RefineConfig dataclass with pipeline hyperparameters"
```

---

### Task 3: Create stub pipeline and all module files

**Files:**
- Create: `spag4d/refine/pipeline.py`
- Create: `spag4d/refine/camera_rig.py`
- Create: `spag4d/refine/mesh_extract.py`
- Create: `spag4d/refine/gsfixer_adapter.py`
- Create: `spag4d/refine/distill.py`
- Create: `spag4d/refine/provenance.py`
- Create: `spag4d/refine/format_compat.py`

All stubs return plausible fake data so the API/UI can be wired up and tested before real internals are implemented.

- [ ] **Step 1: Create format_compat.py stub**

```python
# spag4d/refine/format_compat.py
"""PLY format conversion between SPAG-4D and GSFix3D."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_gaussians_from_ply(ply_path: str, device: str = "cuda"):
    """Load a SPAG-4D PLY file into GSFix3D's GaussianModel format.

    Returns a GaussianModel instance with all attributes loaded.
    """
    # STUB: returns None until Wave 3
    logger.info(f"[stub] load_gaussians_from_ply({ply_path})")
    return None


def save_gaussians_to_ply(gaussians, output_path: str):
    """Save a GaussianModel back to SPAG-4D PLY format."""
    # STUB: copies input PLY to output path
    logger.info(f"[stub] save_gaussians_to_ply -> {output_path}")
```

- [ ] **Step 2: Create camera_rig.py stub**

```python
# spag4d/refine/camera_rig.py
"""Phase 1: Camera placement, rendering, and hole detection."""

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """Perspective camera for novel-view rendering."""
    position: np.ndarray      # [3] world position
    look_at: np.ndarray       # [3] target point
    up: np.ndarray            # [3] up vector
    fov_deg: float            # vertical FOV in degrees
    width: int                # render width
    height: int               # render height

    @property
    def intrinsics(self):
        """Return 3x3 intrinsic matrix."""
        f = self.height / (2 * np.tan(np.radians(self.fov_deg) / 2))
        cx, cy = self.width / 2, self.height / 2
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def generate_camera_rig(
    origin: np.ndarray,
    depth_map: np.ndarray,
    num_directions: int = 12,
    num_depths: int = 3,
    fov_deg: float = 60.0,
    translation_fracs: tuple = (0.05, 0.15, 0.30),
    resolution: int = 512,
) -> list:
    """Generate novel-view cameras that expose disocclusion holes.

    Translates away from origin along azimuthal directions at
    fractions of median scene depth, looking back toward origin.
    """
    # STUB: returns simple orbit cameras
    logger.info(f"[stub] generate_camera_rig: {num_directions} dirs x {num_depths} depths")
    cameras = []
    median_depth = float(np.median(depth_map[depth_map > 0]))

    for azi_idx in range(num_directions):
        azimuth = (2 * np.pi * azi_idx) / num_directions
        for frac in translation_fracs:
            t = frac * median_depth
            cam_pos = origin + t * np.array([
                np.cos(azimuth), 0.0, np.sin(azimuth)
            ])
            cam = CameraPose(
                position=cam_pos,
                look_at=origin.copy(),
                up=np.array([0.0, 1.0, 0.0]),
                fov_deg=fov_deg,
                width=resolution,
                height=resolution,
            )
            cameras.append(cam)

    return cameras


def render_with_hole_mask(gaussians, camera, alpha_threshold=0.1):
    """Render a camera view and extract hole mask.

    Returns:
        rgb: (H, W, 3) float32 rendered image
        hole_mask: (H, W) float32 binary mask (1 = hole)
    """
    # STUB: returns synthetic data
    logger.info("[stub] render_with_hole_mask")
    h, w = camera.height, camera.width
    rgb = np.random.rand(h, w, 3).astype(np.float32) * 0.5 + 0.25
    hole_mask = np.zeros((h, w), dtype=np.float32)
    # Fake some holes in corners
    hole_mask[:64, :64] = 1.0
    hole_mask[-64:, -64:] = 1.0
    return rgb, hole_mask


def select_repair_cameras(cameras, hole_masks, min_hole_fraction=0.03, max_cameras=20):
    """Filter to cameras with significant hole coverage."""
    scored = []
    for i, mask in enumerate(hole_masks):
        frac = float(mask.mean())
        if frac >= min_hole_fraction:
            scored.append((i, frac))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:max_cameras]]


def extract_cubemap_views(panorama, depth_map, face_size=512):
    """Extract 6 cubemap face images and corresponding cameras from panorama.

    Returns:
        faces: list of 6 (face_size, face_size, 3) images
        cameras: list of 6 CameraPose instances
    """
    # STUB: returns slices of panorama
    logger.info("[stub] extract_cubemap_views")
    h, w = panorama.shape[:2]
    faces = []
    cameras = []
    directions = [
        ([0, 0, -1], [0, 1, 0]),   # front
        ([0, 0, 1], [0, 1, 0]),    # back
        ([1, 0, 0], [0, 1, 0]),    # right
        ([-1, 0, 0], [0, 1, 0]),   # left
        ([0, 1, 0], [0, 0, 1]),    # up
        ([0, -1, 0], [0, 0, -1]),  # down
    ]
    for look_dir, up_dir in directions:
        face = np.random.rand(face_size, face_size, 3).astype(np.float32)
        faces.append(face)
        cameras.append(CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array(look_dir, dtype=np.float64),
            up=np.array(up_dir, dtype=np.float64),
            fov_deg=90.0,
            width=face_size,
            height=face_size,
        ))
    return faces, cameras
```

- [ ] **Step 3: Create mesh_extract.py stub**

```python
# spag4d/refine/mesh_extract.py
"""Mesh extraction from depth map for GSFixer dual conditioning."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_conditioning_mesh(depth_map, panorama, simplify_ratio=0.1):
    """Extract a rough textured mesh from DAP depth for GSFixer conditioning.

    Args:
        depth_map: (H, W) equirectangular depth in meters
        panorama: (H, W, 3) equirectangular RGB image
        simplify_ratio: fraction of faces to keep after decimation

    Returns:
        mesh: trimesh.Trimesh object (or None if stub)
    """
    # STUB: returns None until Wave 2
    logger.info(f"[stub] extract_conditioning_mesh (simplify={simplify_ratio})")
    return None


def render_mesh(mesh, camera, resolution=(512, 512)):
    """Render mesh from a camera pose for dual conditioning.

    Returns:
        (H, W, 3) float32 rendered image
    """
    # STUB: returns gray image
    logger.info("[stub] render_mesh")
    return np.ones((*resolution, 3), dtype=np.float32) * 0.5
```

- [ ] **Step 4: Create gsfixer_adapter.py stub**

```python
# spag4d/refine/gsfixer_adapter.py
"""Phase 2: GSFixer model loading, fine-tuning, and inference."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class GSFixerAdapter:
    """Wraps GSFix3D's MarigoldGSFixerPipeline for SPAG-4D integration."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        logger.info(f"[stub] GSFixerAdapter created (checkpoint={checkpoint_path})")

    def load(self):
        """Load the pretrained GSFixer model."""
        # STUB: model stays None
        logger.info("[stub] GSFixerAdapter.load()")

    def finetune(self, gs_renders, gt_images, mesh, cameras,
                 train_steps=500, learning_rate=1e-5):
        """Fine-tune GSFixer on this specific scene.

        Args:
            gs_renders: list of (H,W,3) rendered images from 3DGS
            gt_images: list of (H,W,3) ground truth cubemap faces
            mesh: trimesh.Trimesh conditioning mesh
            cameras: list of CameraPose for mesh rendering
            train_steps: number of fine-tuning steps
            learning_rate: optimizer LR
        """
        logger.info(f"[stub] GSFixerAdapter.finetune({train_steps} steps)")

    def infer(self, gs_renders, hole_masks, mesh, cameras,
              num_steps=50, guidance_scale=7.5):
        """Run GSFixer inference on hole-containing renders.

        Args:
            gs_renders: list of (H,W,3) renders with holes
            hole_masks: list of (H,W) binary masks (1=hole)
            mesh: conditioning mesh
            cameras: camera poses for mesh rendering
            num_steps: DDIM denoising steps
            guidance_scale: classifier-free guidance scale

        Returns:
            list of (H,W,3) repaired images
        """
        logger.info(f"[stub] GSFixerAdapter.infer({len(gs_renders)} views)")
        # STUB: return input images with holes filled by average color
        repaired = []
        for img, mask in zip(gs_renders, hole_masks):
            out = img.copy()
            if mask.sum() > 0:
                avg_color = img[mask < 0.5].mean(axis=0) if (mask < 0.5).any() else np.array([0.5, 0.5, 0.5])
                out[mask > 0.5] = avg_color
            repaired.append(out)
        return repaired

    def unload(self):
        """Free GPU memory."""
        self.model = None
        logger.info("[stub] GSFixerAdapter.unload()")
```

- [ ] **Step 5: Create distill.py stub**

```python
# spag4d/refine/distill.py
"""Phase 3: 3DGS distillation — optimize Gaussians against repaired images."""

import logging

logger = logging.getLogger(__name__)


def distill_to_gaussians(
    gaussians,
    repaired_images,
    cameras,
    hole_masks,
    original_images=None,
    original_cameras=None,
    num_iterations=3000,
    densify_interval=100,
    densify_grad_threshold=0.0002,
    lr_position=0.00016,
    lr_feature=0.0025,
    lr_opacity=0.05,
    lr_scaling=0.005,
    lr_rotation=0.001,
    original_view_ratio=0.3,
):
    """Optimize 3DGS to match repaired images via differentiable rendering.

    Key insight: depth alignment is emergent — the optimizer places new
    Gaussians at whatever 3D positions produce correct 2D appearance
    across multiple viewpoints simultaneously.

    Returns:
        Updated GaussianModel
    """
    logger.info(f"[stub] distill_to_gaussians({num_iterations} iters, {len(repaired_images)} views)")
    # STUB: return gaussians unchanged
    return gaussians
```

- [ ] **Step 6: Create provenance.py stub**

```python
# spag4d/refine/provenance.py
"""Gaussian provenance tracking: original vs. refinement-created."""

import logging

logger = logging.getLogger(__name__)


def tag_gaussian_provenance(gaussians, initial_count):
    """Mark which Gaussians are original vs. created during refinement.

    Args:
        gaussians: GaussianModel
        initial_count: number of Gaussians before refinement started

    All Gaussians with index < initial_count are 'original' and get
    reduced learning rates (0.1x) to prevent drift.
    """
    logger.info(f"[stub] tag_gaussian_provenance(initial={initial_count})")


def apply_provenance_lr_scaling(gaussians, optimizer, initial_count, scale=0.1):
    """Reduce LR for original Gaussians to prevent drift."""
    logger.info(f"[stub] apply_provenance_lr_scaling(scale={scale})")
```

- [ ] **Step 7: Create pipeline.py with full orchestration logic (using stubs)**

```python
# spag4d/refine/pipeline.py
"""Top-level refinement pipeline orchestrator."""

import logging
import time
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import RefineConfig
from .camera_rig import (
    generate_camera_rig,
    render_with_hole_mask,
    select_repair_cameras,
    extract_cubemap_views,
)
from .mesh_extract import extract_conditioning_mesh
from .gsfixer_adapter import GSFixerAdapter
from .distill import distill_to_gaussians
from .provenance import tag_gaussian_provenance
from .format_compat import load_gaussians_from_ply, save_gaussians_to_ply

logger = logging.getLogger(__name__)


def refine_splat(
    ply_path: str,
    panorama_path: str,
    depth_map: np.ndarray,
    max_iterations: int = 3,
    num_cameras: int = 36,
    finetune_steps: int = 500,
    output_path: str = None,
    config: RefineConfig = None,
    progress_callback: Optional[Callable] = None,
    diagnostics_dir: Optional[str] = None,
) -> dict:
    """Full GSFix3D refinement pipeline.

    Args:
        ply_path: Path to raw PLY from SPAG-4D core pipeline
        panorama_path: Original equirectangular image path
        depth_map: DAP/DA360 depth output (H, W) numpy array
        max_iterations: Max repair iterations
        num_cameras: Total novel-view cameras to generate
        finetune_steps: GSFixer scene adaptation steps
        output_path: Where to save refined PLY (default: input_refined.ply)
        config: Override default hyperparameters
        progress_callback: fn(round, stage, pct) for UI progress
        diagnostics_dir: Directory to save diagnostic images

    Returns:
        dict with refined_ply_path, initial_hole_fraction, final_hole_fraction,
        gaussians_count, iterations_used, total_time
    """
    config = config or RefineConfig()
    config.max_iterations = max_iterations
    config.finetune_steps = finetune_steps

    start_time = time.time()

    def report(stage, pct, iteration=0):
        logger.info(f"[refine] iter={iteration} stage={stage} pct={pct}")
        if progress_callback:
            progress_callback(iteration, stage, pct)

    # Setup diagnostics
    diag_dir = Path(diagnostics_dir) if diagnostics_dir else None
    if diag_dir:
        diag_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---
    from PIL import Image
    panorama = np.array(Image.open(panorama_path)).astype(np.float32) / 255.0

    report("camera_rig", 5)

    # --- Phase 1: Camera rig ---
    logger.info("[refine] Phase 1: Generating camera rig...")
    num_dirs = max(num_cameras // 3, 4)
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth_map,
        num_directions=num_dirs,
        num_depths=3,
        fov_deg=config.camera_fov,
        translation_fracs=config.translation_fracs,
        resolution=config.render_resolution,
    )

    report("mesh_extract", 10)

    # --- Extract conditioning mesh ---
    logger.info("[refine] Extracting conditioning mesh...")
    mesh = extract_conditioning_mesh(
        depth_map=depth_map,
        panorama=panorama,
        simplify_ratio=config.mesh_simplify_ratio,
    )

    # --- Prepare anchor views ---
    logger.info("[refine] Preparing anchor views from panorama...")
    cubemap_faces, cubemap_cameras = extract_cubemap_views(
        panorama, depth_map, face_size=config.render_resolution,
    )

    report("finetune", 15)

    # --- Phase 2: Load & fine-tune GSFixer ---
    logger.info("[refine] Phase 2: Loading GSFixer...")
    gsfixer = GSFixerAdapter(
        checkpoint_path=config.gsfixer_checkpoint,
        device="cuda",
    )
    gsfixer.load()

    logger.info(f"[refine] Fine-tuning GSFixer ({config.finetune_steps} steps)...")
    gsfixer.finetune(
        gs_renders=cubemap_faces,
        gt_images=cubemap_faces,
        mesh=mesh,
        cameras=cubemap_cameras,
        train_steps=config.finetune_steps,
        learning_rate=config.finetune_lr,
    )

    report("render_holes", 30)

    # --- Load Gaussians ---
    gaussians = load_gaussians_from_ply(ply_path, device="cuda")

    # --- Iterative repair loop ---
    initial_hole_frac = None
    avg_hole_frac = 0.0

    for iteration in range(config.max_iterations):
        iter_num = iteration + 1
        logger.info(f"\n[refine] === Iteration {iter_num}/{config.max_iterations} ===")

        report("render_holes", 30 + iteration * 20, iter_num)

        # Render all cameras, detect holes
        renders, masks = [], []
        for cam in cameras:
            rgb, mask = render_with_hole_mask(
                gaussians, cam, alpha_threshold=config.alpha_threshold,
            )
            renders.append(rgb)
            masks.append(mask)

        avg_hole_frac = float(np.mean([m.mean() for m in masks]))
        if initial_hole_frac is None:
            initial_hole_frac = avg_hole_frac
        logger.info(f"[refine] Average hole fraction: {avg_hole_frac:.4f}")

        if avg_hole_frac < config.convergence_threshold:
            logger.info("[refine] Converged — holes below threshold.")
            break

        # Select views needing repair
        repair_indices = select_repair_cameras(
            cameras, masks,
            min_hole_fraction=config.min_hole_fraction,
            max_cameras=config.max_repair_cameras,
        )
        logger.info(f"[refine] Repairing {len(repair_indices)} views")

        if len(repair_indices) == 0:
            logger.info("[refine] No views need repair — done.")
            break

        report("gsfixer_inference", 50 + iteration * 15, iter_num)

        # GSFixer inference
        repair_renders = [renders[i] for i in repair_indices]
        repair_masks = [masks[i] for i in repair_indices]
        repair_cams = [cameras[i] for i in repair_indices]

        repaired = gsfixer.infer(
            gs_renders=repair_renders,
            hole_masks=repair_masks,
            mesh=mesh,
            cameras=repair_cams,
            num_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
        )

        # Save diagnostics
        if diag_dir:
            _save_diagnostics(diag_dir, iter_num, repair_renders,
                              repair_masks, repaired, repair_indices)

        report("distill", 70 + iteration * 10, iter_num)

        # Phase 3: Distill back into 3DGS
        logger.info(f"[refine] Distilling {len(repaired)} repaired views...")
        gaussians = distill_to_gaussians(
            gaussians=gaussians,
            repaired_images=repaired,
            cameras=repair_cams,
            hole_masks=repair_masks,
            original_images=cubemap_faces,
            original_cameras=cubemap_cameras,
            num_iterations=config.distill_iterations,
            densify_interval=config.densify_interval,
            densify_grad_threshold=config.densify_grad_threshold,
        )

    # Cleanup
    gsfixer.unload()

    # --- Export refined PLY ---
    output_path = output_path or ply_path.replace('.ply', '_refined.ply')
    # STUB: copy original PLY as "refined" until real implementation
    if not Path(output_path).exists():
        shutil.copy2(ply_path, output_path)
    save_gaussians_to_ply(gaussians, output_path)

    report("distill", 100, config.max_iterations)

    total_time = time.time() - start_time
    logger.info(f"\n[refine] Done in {total_time:.1f}s. Holes: {initial_hole_frac:.4f} -> {avg_hole_frac:.4f}")

    return {
        "refined_ply_path": output_path,
        "initial_hole_fraction": initial_hole_frac or 0.0,
        "final_hole_fraction": avg_hole_frac,
        "gaussians_count": 0,  # Updated when real implementation
        "iterations_used": min(iteration + 1, config.max_iterations),
        "total_time": round(total_time, 1),
    }


def _save_diagnostics(diag_dir, iteration, renders, masks, repaired, indices):
    """Save diagnostic images for UI gallery."""
    from PIL import Image

    for i, (render, mask, repair, cam_idx) in enumerate(
        zip(renders, masks, repaired, indices)
    ):
        prefix = f"r{iteration}_cam{cam_idx}"
        # Save render
        img = Image.fromarray((render * 255).clip(0, 255).astype(np.uint8))
        img.save(diag_dir / f"{prefix}_splat.png")
        # Save mask
        mask_img = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8))
        mask_img.save(diag_dir / f"{prefix}_mask.png")
        # Save repaired
        rep_img = Image.fromarray((repair * 255).clip(0, 255).astype(np.uint8))
        rep_img.save(diag_dir / f"{prefix}_repaired.png")
```

- [ ] **Step 8: Update `__init__.py` to expose public API**

```python
# spag4d/refine/__init__.py
"""GSFix3D-based disocclusion repair for panorama-sourced 3DGS."""

from .pipeline import refine_splat
from .config import RefineConfig

__all__ = ["refine_splat", "RefineConfig"]
```

- [ ] **Step 9: Commit all stubs**

```bash
git add spag4d/refine/
git commit -m "feat(refine): scaffold GSFix3D refine module with stub implementations"
```

---

### Task 4: Rewire API refine endpoints to new module

**Files:**
- Modify: `api.py:398-582`

- [ ] **Step 1: Replace `POST /api/refine` endpoint parameters**

In `api.py`, replace the `start_refinement` function (lines ~401-447):

```python
@app.post("/api/refine")
async def start_refinement(
    job_id: str = Query(..., description="Source conversion job ID"),
    max_rounds: int = Query(3, ge=1, le=5),
    num_cameras: int = Query(36, ge=6, le=72),
    finetune_steps: int = Query(500, ge=100, le=2000),
):
    """Start GSFix3D refinement on an existing conversion job."""
    if job_id not in jobs:
        raise HTTPException(404, "Source job not found")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Source job not complete")

    if not (job.output_ply_path and job.output_ply_path.exists()):
        raise HTTPException(400, "PLY file not found")
    if not (job.input_path and job.input_path.exists()):
        raise HTTPException(400, "Input panorama not found (may have been cleaned up)")
    if not (job.depth_npy_path and job.depth_npy_path.exists()):
        raise HTTPException(400, "Depth map not found")

    refine_id = str(uuid.uuid4())
    refine_job = RefineJobInfo(refine_id, job_id)
    refine_job.params = {
        "max_rounds": max_rounds,
        "num_cameras": num_cameras,
        "finetune_steps": finetune_steps,
    }
    refine_job.output_ply_path = TEMP_DIR / f"{refine_id}_refined.ply"
    refine_job.diagnostics_dir = TEMP_DIR / f"{refine_id}_diagnostics"
    refine_jobs[refine_id] = refine_job

    asyncio.create_task(process_refinement(refine_job, job))

    return JSONResponse({
        "refine_job_id": refine_id,
        "status": "queued",
    })
```

- [ ] **Step 2: Replace `_run_refinement` function**

Replace the `_run_refinement` function (lines ~475-582):

```python
def _run_refinement(source_job: JobInfo, refine_job: RefineJobInfo) -> dict:
    """Execute the GSFix3D refinement pipeline (blocking, runs in thread)."""
    import numpy as np
    from spag4d.refine import refine_splat

    params = refine_job.params
    output_dir = refine_job.diagnostics_dir or TEMP_DIR / f"{refine_job.refine_id}_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load depth map
    depth_map = np.load(str(source_job.depth_npy_path))

    def update_progress(round_num, stage, pct):
        refine_job.round_number = round_num
        refine_job.stage = stage
        refine_job.progress_pct = pct
        refine_job.last_updated = time.time()

    result = refine_splat(
        ply_path=str(source_job.output_ply_path),
        panorama_path=str(source_job.input_path),
        depth_map=depth_map,
        max_iterations=params.get("max_rounds", 3),
        num_cameras=params.get("num_cameras", 36),
        finetune_steps=params.get("finetune_steps", 500),
        output_path=str(refine_job.output_ply_path),
        progress_callback=update_progress,
        diagnostics_dir=str(output_dir / "diagnostics"),
    )

    return {
        "initial_hole_fraction": result["initial_hole_fraction"],
        "final_hole_fraction": result["final_hole_fraction"],
        "final_count": result["gaussians_count"],
        "iterations_used": result["iterations_used"],
        "total_time": result["total_time"],
    }
```

- [ ] **Step 3: Remove the heatmap endpoint**

Delete the `download_heatmap_ply` function (lines ~637-649) and remove `heatmap_ply_path` from `RefineJobInfo.__init__`.

- [ ] **Step 4: Clean up RefineJobInfo**

Update `RefineJobInfo.__init__` to remove `heatmap_ply_path`:

```python
class RefineJobInfo:
    """Tracks a refinement job."""

    def __init__(self, refine_id: str, source_job_id: str):
        self.refine_id = refine_id
        self.source_job_id = source_job_id
        self.status = "queued"
        self.created_at = time.time()
        self.last_updated = time.time()
        self.round_number = 0
        self.stage = ""
        self.progress_pct = 0
        self.output_ply_path: Optional[Path] = None
        self.diagnostics_dir: Optional[Path] = None
        self.metrics: dict = {}
        self.error: Optional[str] = None
        self.params: dict = {}
```

- [ ] **Step 5: Update refine status endpoint**

In the `get_refine_status` function, remove heatmap URL references:

```python
@app.get("/api/refine/status/{refine_id}")
async def get_refine_status(refine_id: str):
    """Get refinement job status."""
    if refine_id not in refine_jobs:
        raise HTTPException(404, "Refinement job not found")

    rj = refine_jobs[refine_id]
    response = {
        "refine_job_id": refine_id,
        "status": rj.status,
        "round": rj.round_number,
        "stage": rj.stage,
        "progress_pct": rj.progress_pct,
    }

    if rj.diagnostics_dir and rj.diagnostics_dir.exists():
        response["diagnostics_url"] = f"/api/refine/diagnostics/{refine_id}"

    if rj.status == "complete":
        response["metrics"] = rj.metrics
        if rj.output_ply_path and rj.output_ply_path.exists():
            response["ply_url"] = f"/api/refine/download/{refine_id}"

    if rj.status == "error":
        response["error"] = rj.error

    return JSONResponse(response)
```

- [ ] **Step 6: Commit API changes**

```bash
git add api.py
git commit -m "feat(api): rewire refine endpoints to GSFix3D pipeline, remove Klein refs"
```

---

### Task 5: Update UI — simplify refine panel and stage labels

**Files:**
- Modify: `static/index.html:93-325`
- Modify: `static/js/app.js:384-505`

- [ ] **Step 1: Replace refine panel HTML**

In `static/index.html`, replace the refine panel (lines 253-325) with:

```html
            <!-- Refinement Controls -->
            <div id="refine-panel" class="controls-panel" style="display: none;">
                <div class="actions-row">
                    <span class="section-label">Splat Refinement (GSFix3D)</span>
                    <button id="refine-btn" class="action-btn" disabled>
                        <span class="btn-icon">🔧</span>
                        Refine
                    </button>
                    <button id="download-refined-btn" class="action-btn" disabled>
                        <span class="btn-icon">💾</span>
                        Refined PLY
                    </button>
                </div>
                <div class="params-row">
                    <div class="param-group" title="Number of novel-view cameras for hole detection. More = better coverage, slower.">
                        <label for="num-cameras">Cameras</label>
                        <select id="num-cameras">
                            <option value="12">12</option>
                            <option value="24">24</option>
                            <option value="36" selected>36</option>
                            <option value="48">48</option>
                        </select>
                    </div>
                    <div class="param-group" title="Number of refinement iterations. Each round detects remaining holes and repairs them.">
                        <label for="max-rounds">Rounds</label>
                        <select id="max-rounds">
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3" selected>3</option>
                        </select>
                    </div>
                    <details class="param-group advanced-params">
                        <summary>Advanced</summary>
                        <div class="param-group" title="GSFixer scene adaptation steps. More = better quality, slower (~1min per 100 steps).">
                            <label for="finetune-steps">Finetune Steps</label>
                            <input type="number" id="finetune-steps" value="500" min="100" max="2000" step="100">
                        </div>
                    </details>
                </div>
                <!-- Refinement progress -->
                <div id="refine-status" class="refine-status" style="display: none;">
                    <div class="refine-progress-bar">
                        <div id="refine-progress-fill" class="progress-fill" style="width: 0%"></div>
                    </div>
                    <span id="refine-status-text">Idle</span>
                </div>
                <!-- Metrics display (after completion) -->
                <div id="refine-metrics" class="refine-metrics" style="display: none;"></div>
            </div>
```

- [ ] **Step 2: Update help panel text**

In `static/index.html`, replace the "Splat Refinement" help section (lines 130-156) with:

```html
                <h3>Splat Refinement (GSFix3D)</h3>
                <p>After conversion, use refinement to fill disocclusion holes. The pipeline generates novel-view cameras, detects holes via alpha thresholding, repairs them with a scene-adapted diffusion model (GSFix3D), then optimizes the 3DGS via differentiable rendering.</p>
                <dl class="help-params">
                    <dt>Cameras</dt>
                    <dd>Number of novel-view cameras. 36 = 12 directions x 3 depths. More cameras detect more holes but take longer.</dd>

                    <dt>Rounds</dt>
                    <dd>Refinement iterations. Each round re-renders, finds remaining holes, and repairs them. Stops early when holes &lt; 2%.</dd>

                    <dt>Finetune Steps</dt>
                    <dd>GSFixer scene adaptation. 500 steps (~5 min) teaches the model this scene's appearance before repairing holes.</dd>
                </dl>

                <div class="tip">
                    Quick Tips: Start with 36 cameras and 1 round for a quick test. The pipeline auto-detects which cameras have significant holes and only repairs those. Use 3 rounds for thorough coverage.
                </div>
```

- [ ] **Step 3: Update app.js — simplify startRefinement()**

In `static/js/app.js`, replace the `startRefinement` method (lines 384-423):

```javascript
    async startRefinement() {
        if (!this.currentJobId) return;

        const refineBtn = document.getElementById('refine-btn');
        if (refineBtn) refineBtn.disabled = true;

        const params = new URLSearchParams({
            job_id: this.currentJobId,
            num_cameras: document.getElementById('num-cameras')?.value || '36',
            max_rounds: document.getElementById('max-rounds')?.value || '3',
            finetune_steps: document.getElementById('finetune-steps')?.value || '500',
        });

        const refineStatus = document.getElementById('refine-status');
        if (refineStatus) refineStatus.style.display = '';
        this.setRefineStatus('Starting refinement...', 0);
        const diagBtn = document.getElementById('show-diagnostics-btn');
        if (diagBtn) diagBtn.disabled = false;

        try {
            const response = await fetch(`/api/refine?${params}`, { method: 'POST' });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Refinement failed to start');
            }
            const result = await response.json();
            this.currentRefineId = result.refine_job_id;
            this.startRefinePoll();
        } catch (error) {
            this.setRefineStatus(`Error: ${error.message}`, 0);
            if (refineBtn) refineBtn.disabled = false;
        }
    }
```

- [ ] **Step 4: Update checkRefineStatus() stage labels**

In `static/js/app.js`, update the `checkRefineStatus` method (lines 430-478). Replace the processing label logic:

```javascript
    async checkRefineStatus() {
        if (!this.currentRefineId) return;

        const stageLabels = {
            'camera_rig': 'Generating cameras',
            'mesh_extract': 'Extracting mesh',
            'finetune': 'Adapting to scene',
            'render_holes': 'Detecting holes',
            'gsfixer_inference': 'Repairing holes',
            'distill': 'Optimizing 3D',
        };

        try {
            const response = await fetch(`/api/refine/status/${this.currentRefineId}`);
            if (!response.ok) throw new Error('Status check failed');
            const status = await response.json();

            if (status.status === 'processing') {
                const stageName = stageLabels[status.stage] || status.stage || 'Processing';
                const label = status.round > 0
                    ? `Round ${status.round} — ${stageName}`
                    : stageName;
                this.setRefineStatus(label, status.progress_pct);
            } else if (status.status === 'complete') {
                clearInterval(this.refinePollInterval);
                this.refinePollInterval = null;
                this.setRefineStatus('Refinement complete!', 100);

                this.refinedPlyUrl = status.ply_url || null;

                if (status.ply_url) {
                    this.ensureViewer();
                    this.viewer.loadScene(status.ply_url);
                }

                if (status.metrics) this.showRefineMetrics(status.metrics);

                const dlBtn = document.getElementById('download-refined-btn');
                if (dlBtn) dlBtn.disabled = false;

                const refineBtn = document.getElementById('refine-btn');
                if (refineBtn) refineBtn.disabled = false;

            } else if (status.status === 'error') {
                clearInterval(this.refinePollInterval);
                this.refinePollInterval = null;
                this.setRefineStatus(`Error: ${status.error}`, 0);
                const refineBtn = document.getElementById('refine-btn');
                if (refineBtn) refineBtn.disabled = false;
            }
        } catch (error) {
            console.error('Refine poll error:', error);
        }
    }
```

- [ ] **Step 5: Update showRefineMetrics() for new metric names**

Replace the `showRefineMetrics` method (lines 488-505):

```javascript
    showRefineMetrics(metrics) {
        const container = document.getElementById('refine-metrics');
        if (!container) return;
        container.style.display = '';

        const items = [
            { label: 'Holes Before', value: metrics.initial_hole_fraction != null ? `${(metrics.initial_hole_fraction * 100).toFixed(1)}%` : '—' },
            { label: 'Holes After', value: metrics.final_hole_fraction != null ? `${(metrics.final_hole_fraction * 100).toFixed(1)}%` : '—' },
            { label: 'Gaussians', value: metrics.final_count?.toLocaleString() || '—' },
            { label: 'Rounds', value: metrics.iterations_used || '—' },
            { label: 'Time', value: metrics.total_time ? `${metrics.total_time}s` : '—' },
        ];

        container.innerHTML = items.map(m =>
            `<div class="metric-card"><div class="metric-value">${m.value}</div><div class="metric-label">${m.label}</div></div>`
        ).join('');
    }
```

- [ ] **Step 6: Remove Klein-specific JS code**

Remove these methods and their event listeners from `app.js`:
- `toggleCameraPreset()` (lines 509-520)
- `addCustomCamera()` (lines 522-536)
- `clearCustomCameras()` (lines 538-542)
- `toggleHeatmap()` (lines 546-557)

Remove corresponding event listeners from `init()` (lines 103-123):
- Camera preset listener
- Add/clear camera button listeners
- Heatmap toggle listener

Remove from constructor (line 15): `this.customCameras = []`
Remove from constructor (line 16): `this.heatmapUrl = null`
Remove from constructor (line 18): `this.showingHeatmap = false`

- [ ] **Step 7: Commit UI changes**

```bash
git add static/index.html static/js/app.js
git commit -m "feat(ui): simplify refine panel for GSFix3D, remove Klein controls"
```

---

### Task 6: Test stub pipeline through Chrome UI

**Files:** None (testing only)

- [ ] **Step 1: Start the SPAG-4D server**

```bash
cd D:/SPAG-4D
.venv/Scripts/python -m spag4d serve --port 7860
```

- [ ] **Step 2: Open in Chrome and verify core flow**

Navigate to `http://localhost:7860`. Verify:
- Page loads without JS errors (check console)
- Upload button works
- Convert button triggers conversion
- PLY loads in 3D viewer
- Refine panel appears after conversion completes

- [ ] **Step 3: Test refine flow with stubs**

Click "Refine" and verify:
- Request hits `POST /api/refine` with `num_cameras`, `max_rounds`, `finetune_steps`
- Progress bar updates with new stage labels
- On completion, "refined" PLY loads in viewer (stub: copy of original)
- Metrics display shows hole fractions
- Download button works
- No JS errors in console

- [ ] **Step 4: Verify removed controls are gone**

Confirm these are NOT present:
- Backend selector dropdown
- Orbit radius input
- Camera preset selector
- Custom camera controls
- Heatmap toggle button

- [ ] **Step 5: Commit any fixes from testing**

```bash
git add -A
git commit -m "fix(ui): address issues found during stub pipeline Chrome testing"
```

---

## Wave 2: Phase 1 Internals (Camera Rig + Hole Detection)

### Task 7: Implement format_compat.py — PLY round-trip

**Files:**
- Modify: `spag4d/refine/format_compat.py`
- Test: `tests/test_format_compat.py`

- [ ] **Step 1: Write PLY round-trip test**

```python
# tests/test_format_compat.py
import numpy as np
import tempfile
from pathlib import Path


def test_load_and_save_ply_roundtrip(tmp_path):
    """Load a SPAG-4D PLY, save it back, verify attributes preserved."""
    from spag4d.refine.format_compat import load_gaussians_from_ply, save_gaussians_to_ply

    # Create a minimal synthetic PLY
    ply_path = _create_synthetic_ply(tmp_path / "test.ply", n_gaussians=100)

    # Load
    gaussians = load_gaussians_from_ply(str(ply_path), device="cpu")
    assert gaussians is not None

    # Save
    out_path = tmp_path / "roundtrip.ply"
    save_gaussians_to_ply(gaussians, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Reload and compare
    gaussians2 = load_gaussians_from_ply(str(out_path), device="cpu")
    assert gaussians2 is not None


def _create_synthetic_ply(path, n_gaussians=100):
    """Create a minimal standard 3DGS PLY for testing."""
    import struct

    header = f"""ply
format binary_little_endian 1.0
element vertex {n_gaussians}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    rng = np.random.default_rng(42)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for _ in range(n_gaussians):
            xyz = rng.normal(0, 1, 3).astype(np.float32)
            sh = rng.normal(0, 0.1, 3).astype(np.float32)
            opacity = np.float32(2.0)  # logit
            scale = rng.normal(-3, 0.5, 3).astype(np.float32)  # log scale
            rot = np.array([1, 0, 0, 0], dtype=np.float32)  # wxyz identity
            f.write(struct.pack('<3f', *xyz))
            f.write(struct.pack('<3f', *sh))
            f.write(struct.pack('<f', opacity))
            f.write(struct.pack('<3f', *scale))
            f.write(struct.pack('<4f', *rot))
    return path
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python -m pytest tests/test_format_compat.py -v
```

Expected: FAIL (stub returns None)

- [ ] **Step 3: Implement format_compat.py**

```python
# spag4d/refine/format_compat.py
"""PLY format conversion between SPAG-4D and GSFix3D."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Add GSFix3D to path for GaussianModel import
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


def load_gaussians_from_ply(ply_path: str, device: str = "cuda"):
    """Load a SPAG-4D PLY file into GSFix3D's GaussianModel.

    Both SPAG-4D and GSFix3D use standard 3DGS PLY format:
    - Quaternions: WXYZ in PLY (rot_0=W, rot_1=X, rot_2=Y, rot_3=Z)
    - Opacity: logit-encoded
    - Scale: log-encoded
    - SH: degree-0 (f_dc_0, f_dc_1, f_dc_2)

    Returns:
        GaussianModel instance with loaded attributes
    """
    from gs.gaussian_model import GaussianModel

    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(ply_path)

    if device != "cpu":
        gaussians = gaussians.to(device)

    logger.info(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians from {ply_path}")
    return gaussians


def save_gaussians_to_ply(gaussians, output_path: str):
    """Save GaussianModel back to standard 3DGS PLY format."""
    if gaussians is None:
        logger.warning("save_gaussians_to_ply: gaussians is None, skipping")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(output_path)
    logger.info(f"Saved Gaussians to {output_path}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/Scripts/python -m pytest tests/test_format_compat.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/format_compat.py tests/test_format_compat.py
git commit -m "feat(refine): implement PLY format round-trip via GSFix3D GaussianModel"
```

---

### Task 8: Implement camera_rig.py — real rendering and hole detection

**Files:**
- Modify: `spag4d/refine/camera_rig.py`
- Test: `tests/test_camera_rig.py`

- [ ] **Step 1: Write camera geometry test**

```python
# tests/test_camera_rig.py
import numpy as np
from spag4d.refine.camera_rig import generate_camera_rig, CameraPose, select_repair_cameras


def test_camera_rig_count():
    """36 cameras: 12 directions x 3 depths."""
    depth = np.ones((180, 360), dtype=np.float32) * 5.0
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth,
        num_directions=12,
        num_depths=3,
    )
    assert len(cameras) == 36


def test_cameras_translate_away_from_origin():
    """All cameras should be displaced from origin."""
    depth = np.ones((180, 360), dtype=np.float32) * 10.0
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth,
        num_directions=4,
        num_depths=2,
    )
    for cam in cameras:
        dist = np.linalg.norm(cam.position)
        assert dist > 0.01, "Camera should be displaced from origin"


def test_cameras_look_back_toward_origin():
    """Camera look_at should point toward origin."""
    depth = np.ones((180, 360), dtype=np.float32) * 5.0
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth,
        num_directions=4,
        num_depths=1,
    )
    for cam in cameras:
        np.testing.assert_array_equal(cam.look_at, [0.0, 0.0, 0.0])


def test_intrinsics_matrix():
    """Verify intrinsic matrix computation."""
    cam = CameraPose(
        position=np.zeros(3),
        look_at=np.array([0, 0, -1.0]),
        up=np.array([0, 1.0, 0]),
        fov_deg=90.0,
        width=512,
        height=512,
    )
    K = cam.intrinsics
    assert K.shape == (3, 3)
    assert abs(K[0, 0] - 256.0) < 1.0  # f = h/(2*tan(45)) = 256
    assert abs(K[0, 2] - 256.0) < 0.01  # cx = w/2


def test_select_repair_cameras():
    """Only cameras with sufficient holes should be selected."""
    cameras = [None] * 5
    masks = [
        np.ones((10, 10)) * 0.5,   # 50% holes
        np.ones((10, 10)) * 0.01,  # 1% holes (below threshold)
        np.ones((10, 10)) * 0.1,   # 10% holes
        np.zeros((10, 10)),         # 0% holes
        np.ones((10, 10)) * 0.04,  # 4% holes
    ]
    selected = select_repair_cameras(cameras, masks, min_hole_fraction=0.03, max_cameras=10)
    assert 0 in selected  # 50%
    assert 2 in selected  # 10%
    assert 4 in selected  # 4%
    assert 1 not in selected  # 1% below threshold
    assert 3 not in selected  # 0%
```

- [ ] **Step 2: Run test to verify geometry tests pass (stubs already compute geometry)**

```bash
.venv/Scripts/python -m pytest tests/test_camera_rig.py -v
```

Expected: All geometry tests PASS (the stub already has real geometry logic). The rendering test would need GPU.

- [ ] **Step 3: Implement real rendering in camera_rig.py**

Replace the `render_with_hole_mask` stub and `extract_cubemap_views` with real implementations using `diff-gaussian-rasterization`:

```python
# Add to camera_rig.py — replace render_with_hole_mask stub

def render_with_hole_mask(gaussians, camera, alpha_threshold=0.1):
    """Render a camera view from the 3DGS and extract hole mask.

    Uses diff-gaussian-rasterization for GPU rendering.

    Returns:
        rgb: (H, W, 3) float32 rendered image
        hole_mask: (H, W) float32 binary mask (1 = hole)
    """
    import torch
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    device = gaussians.get_xyz.device
    h, w = camera.height, camera.width

    # Build view matrix from camera pose
    R, T = _camera_to_RT(camera)
    world_view_transform = _compose_view_matrix(R, T).to(device)
    projection_matrix = _perspective_projection(
        camera.fov_deg, w / h, 0.01, 100.0
    ).to(device)
    full_proj = world_view_transform @ projection_matrix

    campos = torch.tensor(camera.position, dtype=torch.float32, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=np.tan(np.radians(camera.fov_deg) / 2) * (w / h),
        tanfovy=np.tan(np.radians(camera.fov_deg) / 2),
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D[:, :2], device=device)
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    shs = gaussians.get_features

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # rendered_image: (3, H, W) -> (H, W, 3)
    rgb = rendered_image.permute(1, 2, 0).detach().cpu().numpy()

    # For hole detection, render alpha channel
    # Alpha = sum of opacity contributions. Approximate via rendered vs bg.
    # Regions where all channels are near-zero (bg=0) are holes.
    alpha = np.sqrt((rgb ** 2).sum(axis=2))
    hole_mask = (alpha < alpha_threshold).astype(np.float32)

    return rgb, hole_mask


def _camera_to_RT(camera):
    """Convert CameraPose to rotation matrix R and translation T."""
    # Forward = normalize(look_at - position)
    forward = camera.look_at - camera.position
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # Right = forward x up
    right = np.cross(forward, camera.up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Recompute up = right x forward
    up = np.cross(right, forward)

    # R: world -> camera (row vectors are camera axes)
    R = np.stack([right, up, -forward], axis=0)  # (3, 3)

    # T: camera-space translation
    T = -R @ camera.position

    return R.astype(np.float32), T.astype(np.float32)


def _compose_view_matrix(R, T):
    """Compose 4x4 view matrix from R and T."""
    import torch
    view = torch.eye(4)
    view[:3, :3] = torch.from_numpy(R)
    view[:3, 3] = torch.from_numpy(T)
    return view


def _perspective_projection(fov_deg, aspect, near, far):
    """Build 4x4 perspective projection matrix."""
    import torch
    fov_rad = np.radians(fov_deg)
    tan_half_fov = np.tan(fov_rad / 2)

    proj = torch.zeros(4, 4)
    proj[0, 0] = 1.0 / (aspect * tan_half_fov)
    proj[1, 1] = 1.0 / tan_half_fov
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj
```

Note: The exact rasterizer API may need adjustment based on GSFix3D's version of `diff-gaussian-rasterization`. Verify against the installed version's signature.

- [ ] **Step 4: Implement real cubemap extraction**

Replace the `extract_cubemap_views` stub with real equirectangular-to-cubemap sampling:

```python
def extract_cubemap_views(panorama, depth_map, face_size=512):
    """Extract 6 cubemap faces from equirectangular panorama.

    Returns:
        faces: list of 6 (face_size, face_size, 3) float32 images
        cameras: list of 6 CameraPose looking along +-X, +-Y, +-Z
    """
    from scipy.ndimage import map_coordinates

    h, w = panorama.shape[:2]
    faces = []
    cameras = []

    # Cubemap face definitions: (forward_vec, up_vec)
    face_defs = [
        (np.array([0, 0, -1.0]), np.array([0, 1.0, 0])),   # front (-Z)
        (np.array([0, 0, 1.0]),  np.array([0, 1.0, 0])),    # back (+Z)
        (np.array([1, 0, 0.0]),  np.array([0, 1.0, 0])),    # right (+X)
        (np.array([-1, 0, 0.0]), np.array([0, 1.0, 0])),    # left (-X)
        (np.array([0, 1, 0.0]),  np.array([0, 0, -1.0])),   # up (+Y)
        (np.array([0, -1, 0.0]), np.array([0, 0, 1.0])),    # down (-Y)
    ]

    for forward, up in face_defs:
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Generate ray directions for each pixel
        u = np.linspace(-1, 1, face_size)
        v = np.linspace(-1, 1, face_size)
        uu, vv = np.meshgrid(u, v)

        # Ray directions in world space
        dirs = (uu[..., None] * right + vv[..., None] * up + forward)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # Convert to equirectangular coordinates
        theta = np.arctan2(dirs[..., 0], -dirs[..., 2])  # azimuth
        phi = np.arcsin(np.clip(dirs[..., 1], -1, 1))    # elevation

        # Map to pixel coordinates
        px = ((theta / np.pi + 1) / 2) * (w - 1)
        py = ((0.5 - phi / np.pi)) * (h - 1)

        # Sample panorama
        face = np.zeros((face_size, face_size, 3), dtype=np.float32)
        for c in range(3):
            face[..., c] = map_coordinates(
                panorama[..., c], [py, px], order=1, mode='wrap'
            )

        faces.append(face)
        cameras.append(CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=forward.astype(np.float64),
            up=up.astype(np.float64),
            fov_deg=90.0,
            width=face_size,
            height=face_size,
        ))

    return faces, cameras
```

- [ ] **Step 5: Run tests**

```bash
.venv/Scripts/python -m pytest tests/test_camera_rig.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add spag4d/refine/camera_rig.py tests/test_camera_rig.py
git commit -m "feat(refine): implement camera rig generation and hole detection rendering"
```

---

### Task 9: Implement mesh_extract.py

**Files:**
- Modify: `spag4d/refine/mesh_extract.py`
- Test: `tests/test_mesh_extract.py`

- [ ] **Step 1: Write mesh extraction test**

```python
# tests/test_mesh_extract.py
import numpy as np


def test_extract_mesh_from_synthetic_depth():
    """Mesh extraction from a flat floor depth map produces a mesh."""
    from spag4d.refine.mesh_extract import extract_conditioning_mesh

    # Synthetic equirectangular depth: flat floor at 3m
    h, w = 64, 128
    depth = np.ones((h, w), dtype=np.float32) * 3.0
    panorama = np.random.rand(h, w, 3).astype(np.float32)

    mesh = extract_conditioning_mesh(depth, panorama, simplify_ratio=0.5)
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python -m pytest tests/test_mesh_extract.py -v
```

Expected: FAIL (stub returns None)

- [ ] **Step 3: Implement mesh_extract.py**

```python
# spag4d/refine/mesh_extract.py
"""Mesh extraction from depth map for GSFixer dual conditioning."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_conditioning_mesh(depth_map, panorama, simplify_ratio=0.1):
    """Extract a rough textured mesh from equirectangular depth.

    Uses Open3D for Poisson surface reconstruction, then simplifies
    with quadric decimation. The mesh is a conditioning signal for
    GSFixer's dual-input branch — it needs approximate structure,
    not reconstruction-quality geometry.

    Args:
        depth_map: (H, W) equirectangular depth in meters
        panorama: (H, W, 3) equirectangular RGB (float32, 0-1)
        simplify_ratio: fraction of faces to keep

    Returns:
        trimesh.Trimesh with vertex colors
    """
    import open3d as o3d
    import trimesh

    h, w = depth_map.shape

    # Generate equirectangular ray directions
    theta = np.linspace(0, 2 * np.pi, w, endpoint=False)  # azimuth
    phi = np.linspace(np.pi / 2, -np.pi / 2, h)            # elevation

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Spherical to Cartesian
    x = depth_map * np.cos(phi_grid) * np.sin(theta_grid)
    y = depth_map * np.sin(phi_grid)
    z = -depth_map * np.cos(phi_grid) * np.cos(theta_grid)

    # Subsample for speed (every 4th pixel)
    stride = max(1, min(h, w) // 64)
    xs = x[::stride, ::stride].flatten()
    ys = y[::stride, ::stride].flatten()
    zs = z[::stride, ::stride].flatten()
    colors = panorama[::stride, ::stride].reshape(-1, 3)

    # Filter invalid depths
    valid = (depth_map[::stride, ::stride].flatten() > 0.01) & \
            (depth_map[::stride, ::stride].flatten() < 500)
    points = np.stack([xs[valid], ys[valid], zs[valid]], axis=1)
    colors = colors[valid]

    if len(points) < 100:
        logger.warning("Too few valid depth points for mesh extraction")
        return None

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.clip(0, 1))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

    # Poisson reconstruction
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=6
    )

    # Remove low-density vertices (noise)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vertex_colors = np.asarray(mesh_o3d.vertex_colors) if mesh_o3d.has_vertex_colors() else None

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(vertex_colors * 255).astype(np.uint8) if vertex_colors is not None else None,
    )

    # Simplify
    target_faces = max(100, int(len(mesh.faces) * simplify_ratio))
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    logger.info(f"Extracted mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def render_mesh(mesh, camera, resolution=(512, 512)):
    """Render mesh from a camera pose for dual conditioning.

    Uses trimesh's built-in renderer (pyrender backend).

    Returns:
        (H, W, 3) float32 rendered image
    """
    if mesh is None:
        return np.ones((*resolution, 3), dtype=np.float32) * 0.5

    import trimesh

    scene = trimesh.Scene(mesh)

    # Set camera
    from .camera_rig import _camera_to_RT
    R, T = _camera_to_RT(camera)
    transform = np.eye(4)
    transform[:3, :3] = R.T
    transform[:3, 3] = camera.position

    try:
        # Attempt pyrender-based rendering
        data = scene.save_image(resolution=resolution)
        from PIL import Image
        import io
        img = np.array(Image.open(io.BytesIO(data))).astype(np.float32) / 255.0
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    except Exception as e:
        logger.warning(f"Mesh rendering failed ({e}), returning gray placeholder")
        return np.ones((*resolution, 3), dtype=np.float32) * 0.5
```

- [ ] **Step 4: Run test**

```bash
.venv/Scripts/python -m pytest tests/test_mesh_extract.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/mesh_extract.py tests/test_mesh_extract.py
git commit -m "feat(refine): implement depth-to-mesh extraction for dual conditioning"
```

---

## Wave 3: Phase 2 Internals (GSFixer)

### Task 10: Implement gsfixer_adapter.py — model loading and inference

**Files:**
- Modify: `spag4d/refine/gsfixer_adapter.py`

This is the heaviest task. The adapter wraps GSFix3D's `MarigoldGSFixerPipeline` for SPAG-4D's use case.

- [ ] **Step 1: Implement GSFixerAdapter with real model loading**

```python
# spag4d/refine/gsfixer_adapter.py
"""Phase 2: GSFixer model loading, fine-tuning, and inference."""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Add GSFix3D to path
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


class GSFixerAdapter:
    """Wraps GSFix3D's MarigoldGSFixerPipeline for SPAG-4D integration."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.pipe = None

    def load(self):
        """Load the pretrained GSFixer diffusion pipeline."""
        from marigold.pipeline import MarigoldGSFixerPipeline

        logger.info(f"Loading GSFixer from {self.checkpoint_path}...")
        self.pipe = MarigoldGSFixerPipeline.from_pretrained(
            self.checkpoint_path,
        )
        self.pipe = self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception:
            logger.info("xformers not available, using default attention")

        logger.info("GSFixer loaded successfully")

    def finetune(self, gs_renders, gt_images, mesh, cameras,
                 train_steps=500, learning_rate=1e-5):
        """Fine-tune GSFixer on this specific scene.

        Uses cubemap GT views as training targets with the corresponding
        GS renders as inputs. Random mask augmentation teaches the model
        to inpaint within this scene's visual style.
        """
        if self.pipe is None:
            raise RuntimeError("Call load() before finetune()")

        logger.info(f"Fine-tuning GSFixer for {train_steps} steps...")

        # Prepare training data: GS render -> GT pairs
        from torch.utils.data import Dataset, DataLoader

        class SceneDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                inp = torch.from_numpy(self.inputs[idx]).permute(2, 0, 1)  # CHW
                tgt = torch.from_numpy(self.targets[idx]).permute(2, 0, 1)
                # Random mask augmentation
                mask = torch.zeros(1, inp.shape[1], inp.shape[2])
                if torch.rand(1) > 0.3:
                    # Random rectangular mask
                    mh, mw = inp.shape[1], inp.shape[2]
                    y1 = torch.randint(0, mh // 2, (1,)).item()
                    x1 = torch.randint(0, mw // 2, (1,)).item()
                    y2 = y1 + torch.randint(mh // 4, mh // 2, (1,)).item()
                    x2 = x1 + torch.randint(mw // 4, mw // 2, (1,)).item()
                    mask[0, y1:y2, x1:x2] = 1.0
                return inp, tgt, mask

        dataset = SceneDataset(gs_renders, gt_images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Fine-tune UNet only
        unet = self.pipe.unet
        unet.train()
        optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

        for step in range(train_steps):
            for inp, tgt, mask in dataloader:
                inp = inp.to(self.device)
                tgt = tgt.to(self.device)
                mask = mask.to(self.device)

                # Encode to latent space
                with torch.no_grad():
                    latent_tgt = self.pipe.vae.encode(tgt * 2 - 1).latent_dist.sample()
                    latent_tgt = latent_tgt * self.pipe.vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latent_tgt)
                timestep = torch.randint(0, 1000, (1,), device=self.device)
                noisy_latent = self.pipe.scheduler.add_noise(latent_tgt, noise, timestep)

                # Encode input as conditioning
                with torch.no_grad():
                    latent_inp = self.pipe.vae.encode(inp * 2 - 1).latent_dist.sample()
                    latent_inp = latent_inp * self.pipe.vae.config.scaling_factor

                # Concatenate conditioning
                model_input = torch.cat([noisy_latent, latent_inp], dim=1)

                # Predict noise
                noise_pred = unet(model_input, timestep, return_dict=False)[0]

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if step % 100 == 0:
                    logger.info(f"  finetune step {step}/{train_steps}, loss={loss.item():.4f}")
                break  # One batch per step

        unet.eval()
        logger.info("Fine-tuning complete")

    def infer(self, gs_renders, hole_masks, mesh, cameras,
              num_steps=50, guidance_scale=7.5):
        """Run GSFixer inference on hole-containing renders.

        Returns list of repaired (H, W, 3) float32 images.
        """
        if self.pipe is None:
            raise RuntimeError("Call load() before infer()")

        logger.info(f"Running GSFixer inference on {len(gs_renders)} views...")
        repaired = []

        from .mesh_extract import render_mesh

        for i, (gs_img, mask, cam) in enumerate(zip(gs_renders, hole_masks, cameras)):
            logger.info(f"  Repairing view {i+1}/{len(gs_renders)} "
                        f"(hole fraction: {mask.mean():.3f})")

            # Prepare inputs as tensors
            gs_tensor = torch.from_numpy(gs_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)

            # Render mesh for dual conditioning (if available)
            mesh_img = render_mesh(mesh, cam, resolution=(gs_img.shape[0], gs_img.shape[1]))
            mesh_tensor = torch.from_numpy(mesh_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Run diffusion
            with torch.no_grad():
                # Encode GS render
                gs_latent = self.pipe.vae.encode(gs_tensor * 2 - 1).latent_dist.sample()
                gs_latent = gs_latent * self.pipe.vae.config.scaling_factor

                # Start from noise
                latent = torch.randn_like(gs_latent)

                # DDIM denoising
                self.pipe.scheduler.set_timesteps(num_steps)
                for t in self.pipe.scheduler.timesteps:
                    model_input = torch.cat([latent, gs_latent], dim=1)
                    noise_pred = self.pipe.unet(model_input, t, return_dict=False)[0]
                    latent = self.pipe.scheduler.step(noise_pred, t, latent).prev_sample

                # Decode
                image = self.pipe.vae.decode(latent / self.pipe.vae.config.scaling_factor).sample
                image = (image + 1) / 2  # [-1,1] -> [0,1]

            # Composite: keep original content where no holes, use repaired where holes
            result = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = np.clip(result, 0, 1)

            # Blend: original where mask=0, repaired where mask=1
            mask_3d = mask[..., None]
            composited = gs_img * (1 - mask_3d) + result * mask_3d
            repaired.append(composited.astype(np.float32))

        logger.info("GSFixer inference complete")
        return repaired

    def unload(self):
        """Free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logger.info("GSFixer unloaded, GPU memory freed")
```

- [ ] **Step 2: Test model loading (requires checkpoint)**

```bash
.venv/Scripts/python -c "
from spag4d.refine.gsfixer_adapter import GSFixerAdapter
adapter = GSFixerAdapter('goldoak1421/gsfixer-full-replica-room1')
adapter.load()
print('Model loaded successfully')
adapter.unload()
print('Model unloaded')
"
```

Expected: Model downloads from HuggingFace and loads. If this fails due to SD v2 availability, we'll need to source the checkpoint differently.

- [ ] **Step 3: Commit**

```bash
git add spag4d/refine/gsfixer_adapter.py
git commit -m "feat(refine): implement GSFixer adapter with load, finetune, and inference"
```

---

### Task 11: Add GSFix3D checkpoint download to CLI

**Files:**
- Modify: `spag4d/cli.py:129-162`

- [ ] **Step 1: Add gsfix3d to download-models command**

Add a new option and handler to the existing `download_models` function:

```python
@main.command('download-models')
@click.option('--model', type=click.Choice(['dap', 'da360', 'gsfix3d', 'all']),
              default='all', help='Which model weights to download')
@click.option('--verify', is_flag=True, help='Verify downloaded weights')
def download_models(model: str, verify: bool):
    """Download and cache model weights."""
    if model in ('dap', 'all'):
        from .dap_model import DAPModel
        click.echo("Downloading DAP model weights...")
        try:
            path = DAPModel._get_or_download_weights()
            click.echo(f"DAP weights cached at: {path}")
            if verify:
                if DAPModel._verify_checksum(Path(path)):
                    click.echo("Checksum verified")
                else:
                    click.echo("Checksum verification skipped (no reference hash)")
        except Exception as e:
            click.echo(f"DAP download failed: {e}", err=True)
            if model == 'dap':
                raise click.Abort()

    if model in ('da360', 'all'):
        try:
            from .da360_model import DA360Model
            click.echo("Downloading DA360 model weights...")
            path = DA360Model._get_or_download_weights()
            click.echo(f"DA360 weights cached at: {path}")
        except ImportError:
            click.echo("DA360 model not yet available (architecture files needed)", err=True)
        except Exception as e:
            click.echo(f"DA360 download failed: {e}", err=True)
            if model == 'da360':
                raise click.Abort()

    if model in ('gsfix3d', 'all'):
        click.echo("Downloading GSFix3D checkpoint...")
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                "goldoak1421/gsfixer-full-replica-room1",
                local_dir="pretrained/gsfix3d",
            )
            click.echo(f"GSFix3D checkpoint cached at: {path}")
        except ImportError:
            click.echo("huggingface_hub not installed. Install with: pip install huggingface-hub", err=True)
        except Exception as e:
            click.echo(f"GSFix3D download failed: {e}", err=True)
            if model == 'gsfix3d':
                raise click.Abort()
```

- [ ] **Step 2: Test download command**

```bash
.venv/Scripts/python -m spag4d download-models --model gsfix3d
```

Expected: Checkpoint downloads to `pretrained/gsfix3d/`.

- [ ] **Step 3: Commit**

```bash
git add spag4d/cli.py
git commit -m "feat(cli): add GSFix3D checkpoint download to download-models command"
```

---

## Wave 4: Phase 3 Internals (Distillation)

### Task 12: Implement distill.py — 3DGS optimization loop

**Files:**
- Modify: `spag4d/refine/distill.py`

- [ ] **Step 1: Implement the optimization loop**

```python
# spag4d/refine/distill.py
"""Phase 3: 3DGS distillation — optimize Gaussians against repaired images."""

import logging
import random

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def distill_to_gaussians(
    gaussians,
    repaired_images,
    cameras,
    hole_masks,
    original_images=None,
    original_cameras=None,
    num_iterations=3000,
    densify_interval=100,
    densify_grad_threshold=0.0002,
    lr_position=0.00016,
    lr_feature=0.0025,
    lr_opacity=0.05,
    lr_scaling=0.005,
    lr_rotation=0.001,
    original_view_ratio=0.3,
):
    """Optimize 3DGS to match repaired images via differentiable rendering.

    Depth alignment is emergent: the optimizer places new Gaussians at
    whatever 3D positions produce correct 2D appearance across multiple
    viewpoints simultaneously.
    """
    if gaussians is None:
        logger.warning("distill_to_gaussians: gaussians is None, skipping")
        return gaussians

    from .camera_rig import _camera_to_RT, _compose_view_matrix, _perspective_projection
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    device = gaussians.get_xyz.device

    # Convert images to tensors
    repair_tensors = [
        torch.from_numpy(img).permute(2, 0, 1).to(device) for img in repaired_images
    ]
    mask_tensors = [
        torch.from_numpy(m).to(device) for m in hole_masks
    ]

    orig_tensors = None
    if original_images:
        orig_tensors = [
            torch.from_numpy(img).permute(2, 0, 1).to(device) for img in original_images
        ]

    # Setup optimizer
    gaussians.training_setup({
        'position_lr_init': lr_position,
        'position_lr_final': lr_position * 0.01,
        'position_lr_max_steps': num_iterations,
        'feature_lr': lr_feature,
        'opacity_lr': lr_opacity,
        'scaling_lr': lr_scaling,
        'rotation_lr': lr_rotation,
    })

    logger.info(f"Starting distillation: {num_iterations} iterations, "
                f"{len(repaired_images)} repaired + "
                f"{len(original_images) if original_images else 0} original views")

    def _render_differentiable(gaussians, camera):
        """Render using diff-gaussian-rasterization with gradient flow."""
        h, w = camera.height, camera.width
        R, T = _camera_to_RT(camera)
        world_view_transform = _compose_view_matrix(R, T).to(device)
        projection_matrix = _perspective_projection(
            camera.fov_deg, w / h, 0.01, 100.0
        ).to(device)
        full_proj = world_view_transform @ projection_matrix
        campos = torch.tensor(camera.position, dtype=torch.float32, device=device)

        raster_settings = GaussianRasterizationSettings(
            image_height=h, image_width=w,
            tanfovx=np.tan(np.radians(camera.fov_deg) / 2) * (w / h),
            tanfovy=np.tan(np.radians(camera.fov_deg) / 2),
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj,
            sh_degree=0, campos=campos,
            prefiltered=False, debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means2D = torch.zeros_like(gaussians.get_xyz[:, :2], requires_grad=True)

        rendered_image, radii = rasterizer(
            means3D=gaussians.get_xyz,
            means2D=means2D,
            shs=gaussians.get_features,
            colors_precomp=None,
            opacities=gaussians.get_opacity,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            cov3D_precomp=None,
        )
        return rendered_image, means2D, radii  # (3,H,W), grad-enabled

    for step in range(num_iterations):
        # Choose training view
        use_original = (
            orig_tensors is not None
            and random.random() < original_view_ratio
        )

        if use_original:
            idx = random.randint(0, len(orig_tensors) - 1)
            target = orig_tensors[idx]
            camera = original_cameras[idx]
            mask_weight = None
        else:
            idx = random.randint(0, len(repair_tensors) - 1)
            target = repair_tensors[idx]
            camera = cameras[idx]
            # Weight hole regions 2x higher
            mask_weight = mask_tensors[idx] * 2.0 + 1.0

        # Differentiable forward render (gradients flow to Gaussians)
        rendered, means2D, radii = _render_differentiable(gaussians, camera)

        # L1 + D-SSIM loss
        l1_loss = F.l1_loss(rendered, target, reduction='none')
        ssim_loss = 1.0 - _ssim(rendered.unsqueeze(0), target.unsqueeze(0))

        loss = 0.8 * l1_loss.mean() + 0.2 * ssim_loss

        if mask_weight is not None:
            weighted = l1_loss * mask_weight.unsqueeze(0)
            loss = 0.8 * weighted.mean() + 0.2 * ssim_loss

        loss.backward()

        # Accumulate densification stats
        visibility_filter = radii > 0
        gaussians.add_densification_stats(means2D, visibility_filter)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Densification
        if step % densify_interval == 0 and step < num_iterations * 0.8:
            gaussians.densify_and_prune(
                densify_grad_threshold,
                min_opacity=0.005,
                extent=1.0,
                max_screen_size=20,
            )

        if step % 500 == 0:
            logger.info(f"  distill step {step}/{num_iterations}, loss={loss.item():.4f}")

    logger.info(f"Distillation complete. Final Gaussians: {gaussians.get_xyz.shape[0]}")
    return gaussians


def _ssim(img1, img2, window_size=11):
    """Compute structural similarity index (simplified)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()
```

- [ ] **Step 2: Commit**

```bash
git add spag4d/refine/distill.py
git commit -m "feat(refine): implement 3DGS distillation with L1+SSIM loss and densification"
```

---

### Task 13: Implement provenance.py

**Files:**
- Modify: `spag4d/refine/provenance.py`

- [ ] **Step 1: Implement provenance tracking**

```python
# spag4d/refine/provenance.py
"""Gaussian provenance tracking: original vs. refinement-created."""

import logging
import torch

logger = logging.getLogger(__name__)


def tag_gaussian_provenance(gaussians, initial_count):
    """Mark which Gaussians are original vs. created during refinement.

    Stores a provenance tensor on the GaussianModel:
    0 = original (from panorama), 1 = new (from refinement densification)
    """
    if gaussians is None:
        return

    current_count = gaussians.get_xyz.shape[0]
    provenance = torch.zeros(current_count, device=gaussians.get_xyz.device)
    provenance[initial_count:] = 1.0
    gaussians._provenance = provenance
    logger.info(f"Tagged provenance: {initial_count} original, "
                f"{current_count - initial_count} new")


def apply_provenance_lr_scaling(gaussians, initial_count, scale=0.1):
    """Reduce learning rate for original Gaussians to prevent drift.

    Original Gaussians (index < initial_count) get LR scaled by `scale`
    (default 0.1x) while new Gaussians keep full LR.
    """
    if gaussians is None or not hasattr(gaussians, 'optimizer'):
        return

    for param_group in gaussians.optimizer.param_groups:
        if len(param_group['params']) > 0:
            param = param_group['params'][0]
            if param.shape[0] > initial_count:
                # Create per-parameter LR tensor
                lr = param_group['lr']
                lr_tensor = torch.ones(param.shape[0], device=param.device) * lr
                lr_tensor[:initial_count] *= scale
                param_group['lr_per_param'] = lr_tensor

    logger.info(f"Applied {scale}x LR scaling to {initial_count} original Gaussians")
```

- [ ] **Step 2: Update pipeline.py to use provenance**

In `spag4d/refine/pipeline.py`, add provenance tagging after loading Gaussians. Before the iterative loop, add:

```python
    # Tag initial Gaussian count for provenance
    initial_gaussian_count = 0  # Will be set when real loading works
    if gaussians is not None and hasattr(gaussians, 'get_xyz'):
        initial_gaussian_count = gaussians.get_xyz.shape[0]
```

And after distillation in the loop, add:

```python
        # Update provenance tags
        tag_gaussian_provenance(gaussians, initial_gaussian_count)
```

- [ ] **Step 3: Commit**

```bash
git add spag4d/refine/provenance.py spag4d/refine/pipeline.py
git commit -m "feat(refine): implement Gaussian provenance tracking and LR scaling"
```

---

## Wave 5: Integration & Cleanup

### Task 14: Remove pipeline stubs and wire real implementations

**Files:**
- Modify: `spag4d/refine/pipeline.py`

- [ ] **Step 1: Update pipeline.py to remove stub workarounds**

The pipeline.py from Task 3 already calls the real module functions. The key change is removing the stub fallback in the PLY export section. Replace the shutil.copy2 workaround:

```python
    # --- Export refined PLY ---
    output_path = output_path or ply_path.replace('.ply', '_refined.ply')
    save_gaussians_to_ply(gaussians, output_path)
```

Also update the gaussians_count in the return dict:

```python
    gaussian_count = 0
    if gaussians is not None and hasattr(gaussians, 'get_xyz'):
        gaussian_count = gaussians.get_xyz.shape[0]

    return {
        "refined_ply_path": output_path,
        "initial_hole_fraction": initial_hole_frac or 0.0,
        "final_hole_fraction": avg_hole_frac,
        "gaussians_count": gaussian_count,
        "iterations_used": min(iteration + 1, config.max_iterations),
        "total_time": round(total_time, 1),
    }
```

- [ ] **Step 2: Commit**

```bash
git add spag4d/refine/pipeline.py
git commit -m "fix(refine): remove stub workarounds from pipeline, wire real implementations"
```

---

### Task 15: End-to-end test on real panorama

**Files:** None (testing only)

- [ ] **Step 1: Run full pipeline via CLI**

```bash
cd D:/SPAG-4D
.venv/Scripts/python -c "
import numpy as np
from spag4d.refine import refine_splat

# Use an existing PLY + panorama + depth from a previous conversion
result = refine_splat(
    ply_path='output/jobs/test_output.ply',
    panorama_path='TestImage/monbachtal_riverbank_primary.jpg',
    depth_map=np.load('output/jobs/test_depth.npy'),
    max_iterations=1,
    num_cameras=12,
    finetune_steps=100,  # Quick test
    output_path='output/test_refined.ply',
    diagnostics_dir='output/test_diagnostics',
)
print(result)
"
```

Expected: Pipeline runs, outputs refined PLY and diagnostic images. Check logs for errors.

- [ ] **Step 2: Test through Chrome UI**

1. Start server: `.venv/Scripts/python -m spag4d serve`
2. Open `http://localhost:7860`
3. Upload panorama, convert
4. Click Refine (with 12 cameras, 1 round for speed)
5. Verify progress updates, completion, PLY loads, metrics display
6. Check diagnostics gallery shows splat/mask/repaired images

- [ ] **Step 3: Fix any issues found, commit**

```bash
git add -A
git commit -m "fix: address integration issues found during end-to-end testing"
```

---

### Task 16: Delete spag4d-refine/ (Klein removal)

**Files:**
- Delete: `spag4d-refine/` (entire directory)

- [ ] **Step 1: Verify no imports remain**

```bash
cd D:/SPAG-4D
grep -r "spag4d_refine" --include="*.py" --exclude-dir=spag4d-refine | grep -v "^Binary"
grep -r "from spag4d_refine" --include="*.py" --exclude-dir=spag4d-refine
grep -r "import spag4d_refine" --include="*.py" --exclude-dir=spag4d-refine
```

Expected: No matches outside `spag4d-refine/` itself.

- [ ] **Step 2: Delete the directory**

```bash
rm -rf spag4d-refine/
```

- [ ] **Step 3: Clean up any remaining references**

Check and remove from `api.py`:
- Any `sys.path` hacks referencing `spag4d-refine`
- Any unused imports

Check `static/js/app.js` for any remaining Klein references.

- [ ] **Step 4: Verify server still starts**

```bash
.venv/Scripts/python -m spag4d serve --port 7860
```

Open Chrome, verify the full flow still works.

- [ ] **Step 5: Commit deletion**

```bash
git add -A
git commit -m "chore: remove Klein-based spag4d-refine/ — replaced by GSFix3D pipeline"
```

---

### Task 17: Final UI test via Chrome

**Files:** None (testing only)

- [ ] **Step 1: Full flow test**

1. Start server
2. Upload panorama
3. Convert (verify 3D viewer works)
4. Click Refine
5. Verify progress stages show correct labels
6. Verify refined PLY loads in viewer
7. Verify metrics display
8. Verify diagnostics gallery
9. Download refined PLY
10. Check browser console for any JS errors

- [ ] **Step 2: Verify Klein controls are gone**

Confirm none of these exist:
- Backend selector
- Orbit radius input
- Camera preset dropdown
- Custom camera controls
- Heatmap toggle
- Heatmap legend

- [ ] **Step 3: Fix any final issues, commit**

```bash
git add -A
git commit -m "fix: final UI cleanup after Klein removal"
```

---

## Wave 6: CLI Integration

### Task 18: Add --refine flag to convert command

**Files:**
- Modify: `spag4d/cli.py:17-127`

- [ ] **Step 1: Add refine options to convert command**

Add these options to the `@click.option` decorators on the `convert` command:

```python
@click.option('--refine', is_flag=True, default=False,
              help='Run GSFix3D disocclusion repair after conversion')
@click.option('--refine-iterations', default=3, type=int,
              help='Max refinement iterations (default: 3)')
@click.option('--refine-cameras', default=36, type=int,
              help='Novel-view cameras for hole detection (default: 36)')
```

Add to function signature:

```python
def convert(
    input_path, output_path, depth_model, sharp_refine, stride,
    depth_min, depth_max, sky_threshold, outlier_pruning, global_scale,
    sharp_cubemap_size, sharp_projection, force_erp, batch, device,
    quiet, mock_dap,
    refine, refine_iterations, refine_cameras,
):
```

Add after the convert result, before the final output:

```python
    # After conversion completes (both single and batch paths):
    if refine and not batch:
        if not quiet:
            click.echo("Running GSFix3D refinement...")

        from .refine import refine_splat
        import numpy as np

        depth_npy_path = str(output_path).replace('.ply', '_depth.npy')
        if not Path(depth_npy_path).exists():
            click.echo("Warning: depth .npy not found, skipping refinement", err=True)
        else:
            depth_map = np.load(depth_npy_path)
            refined_path = str(output_path).replace('.ply', '_refined.ply')

            refine_result = refine_splat(
                ply_path=str(output_path),
                panorama_path=str(input_path),
                depth_map=depth_map,
                max_iterations=refine_iterations,
                num_cameras=refine_cameras,
                output_path=refined_path,
            )

            if not quiet:
                click.echo(f"Refined: holes {refine_result['initial_hole_fraction']:.2%}"
                           f" -> {refine_result['final_hole_fraction']:.2%}")
                click.echo(f"Saved to: {refined_path}")
```

- [ ] **Step 2: Test CLI refinement**

```bash
.venv/Scripts/python -m spag4d convert TestImage/monbachtal_riverbank_primary.jpg output/test.ply --refine --refine-cameras 12 --refine-iterations 1
```

Expected: Converts, then runs refinement, saves `output/test_refined.ply`.

- [ ] **Step 3: Commit**

```bash
git add spag4d/cli.py
git commit -m "feat(cli): add --refine flag for GSFix3D disocclusion repair"
```
