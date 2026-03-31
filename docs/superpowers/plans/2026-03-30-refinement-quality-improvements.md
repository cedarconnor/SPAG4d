# Refinement Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Klein refinement pipeline auto-adapt to any scene scale (indoor/outdoor), place cameras only where gaps exist, and reject hallucinated synthesis via multi-view color consistency.

**Architecture:** Three independent improvements wired into existing pipeline stages. (1) Scene analysis computes scale-relative defaults after depth estimation. (2) Gap-driven camera selection renders candidate viewpoints cheaply and picks the ones that see disocclusions. (3) Shadow validator extends with color sampling to catch hallucinations.

**Tech Stack:** Python 3.10+, numpy, gsplat (rasterization), existing SPAG-4D pipeline modules.

---

### Task 1: Scene Analysis Module

**Files:**
- Create: `spag4d/scene_analysis.py`
- Test: `tests/test_scene_analysis.py`

- [ ] **Step 1: Write tests for compute_scene_defaults**

```python
# tests/test_scene_analysis.py
import numpy as np
import pytest


def test_outdoor_scene_defaults():
    """Outdoor scene: depths 1-100m, median ~15m."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.lognormal(mean=2.7, sigma=1.0, size=(512, 1024))
    depth = np.clip(depth, 0.5, 200.0)
    result = compute_scene_defaults(depth)

    assert result["sky_threshold"] > 50.0, "Outdoor sky cutoff should be large"
    assert result["depth_min"] > 0.0
    assert result["depth_max"] > result["depth_min"]
    assert result["orbit_radius"] > 0.1
    assert result["orbit_radius"] < 20.0
    assert "confidence_decay_pixels" in result


def test_indoor_scene_defaults():
    """Indoor scene: depths 0.5-5m, median ~2m."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.uniform(0.5, 5.0, size=(512, 1024))
    result = compute_scene_defaults(depth)

    assert result["sky_threshold"] < 10.0, "Indoor sky cutoff should be small"
    assert result["orbit_radius"] < 1.0, "Indoor radius should be small"
    assert result["depth_min"] >= 0.01


def test_auto_parameters_are_positive():
    """All computed parameters must be positive."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.uniform(1.0, 50.0, size=(256, 512))
    result = compute_scene_defaults(depth)

    for key, val in result.items():
        assert val > 0, f"{key} must be positive, got {val}"


def test_handles_zero_depth():
    """Depth maps with zeros (sky/invalid) shouldn't crash."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.uniform(1.0, 20.0, size=(256, 512))
    depth[:50, :] = 0.0  # Sky region
    result = compute_scene_defaults(depth)

    assert result["sky_threshold"] > 0
    assert result["depth_min"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spag4d.scene_analysis'`

- [ ] **Step 3: Implement compute_scene_defaults**

```python
# spag4d/scene_analysis.py
"""Auto-compute scene-relative parameters from depth map statistics."""

from __future__ import annotations

import numpy as np


def compute_scene_defaults(
    depth_map: np.ndarray,
    image_height: int | None = None,
) -> dict:
    """
    Compute scale-relative pipeline defaults from a depth map.

    Works for any scene scale — a 3m room and a 100m forest both
    get appropriate thresholds without manual tuning.

    Args:
        depth_map: [H, W] depth in meters (0 or negative = invalid)
        image_height: Image height for pixel-relative params.
            Defaults to depth_map.shape[0].

    Returns:
        Dict with keys: sky_threshold, depth_min, depth_max,
        orbit_radius, confidence_decay_pixels
    """
    if image_height is None:
        image_height = depth_map.shape[0]

    # Mask out invalid depths (zero, negative, inf)
    valid = (depth_map > 0.01) & np.isfinite(depth_map)
    if valid.sum() < 100:
        # Fallback for degenerate depth maps
        return {
            "sky_threshold": 80.0,
            "depth_min": 0.1,
            "depth_max": 100.0,
            "orbit_radius": 0.5,
            "confidence_decay_pixels": max(10, int(image_height * 0.02)),
        }

    valid_depths = depth_map[valid]

    p1 = float(np.percentile(valid_depths, 1))
    p50 = float(np.percentile(valid_depths, 50))
    p95 = float(np.percentile(valid_depths, 95))
    p99 = float(np.percentile(valid_depths, 99))

    return {
        "sky_threshold": max(p95, p1 + 1.0),  # At least 1m range
        "depth_min": max(0.01, p1 * 0.8),     # 20% margin below 1st percentile
        "depth_max": p99 * 1.1,                # 10% margin above 99th percentile
        "orbit_radius": max(0.05, p50 * 0.05), # 5% of median depth
        "confidence_decay_pixels": max(10, int(image_height * 0.02)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/scene_analysis.py tests/test_scene_analysis.py
git commit -m "feat: add scene_analysis module for scale-relative defaults"
```

---

### Task 2: Wire Scene Analysis into Core Pipeline

**Files:**
- Modify: `spag4d/core.py:69-152` (convert function signature + depth→SPAG gap)
- Modify: `api.py:190-299` (endpoint defaults + pass-through)

- [ ] **Step 1: Update core.py convert() to accept "auto" sentinel and call scene_analysis**

In `spag4d/core.py`, change the convert signature defaults to `None` (meaning "auto"), then resolve after depth estimation:

```python
# In convert() signature, change these defaults:
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    sky_threshold: Optional[float] = None,
    orbit_radius: Optional[float] = None,  # New param for refinement
```

After line 137 (`Depth estimation complete`), before line 147 (`_run_spag_pipeline`), insert:

```python
        # Auto-compute scene-relative defaults for any None parameters
        depth_np = depth.cpu().numpy()
        from .scene_analysis import compute_scene_defaults
        scene_defaults = compute_scene_defaults(depth_np, image_height=H)

        if depth_min is None:
            depth_min = scene_defaults["depth_min"]
        if depth_max is None:
            depth_max = scene_defaults["depth_max"]
        if sky_threshold is None:
            sky_threshold = scene_defaults["sky_threshold"]

        print(f"[SPAG4D] Scene defaults: depth=[{depth_min:.1f}, {depth_max:.1f}]m, "
              f"sky={sky_threshold:.1f}m, orbit_r={scene_defaults['orbit_radius']:.2f}m")
```

- [ ] **Step 2: Update api.py to pass None defaults and return resolved values**

In `api.py`, change the convert endpoint Query defaults:

```python
    depth_min: Optional[float] = Query(None, ge=0.01),
    depth_max: Optional[float] = Query(None, le=1000.0),
    sky_threshold: Optional[float] = Query(None),
```

In the response, add the resolved scene defaults so the UI can display them. After the convert call, store the scene defaults on the job object:

```python
    # After result = await run_in_threadpool(...)
    # Return scene_defaults in the response for UI to display
```

- [ ] **Step 3: Update api.py refine endpoint to use auto orbit_radius**

In the refine endpoint, when `orbit_radius` is not explicitly set, look up the scene defaults from the source conversion job's depth map:

```python
    # In _run_refinement(), after loading depth:
    if params.get("orbit_radius") is None or params.get("orbit_radius") == 0:
        from spag4d.scene_analysis import compute_scene_defaults
        depth_np = np.load(str(source_job.depth_npy_path))
        scene = compute_scene_defaults(depth_np)
        config.orbit_radius = scene["orbit_radius"]
```

- [ ] **Step 4: Test end-to-end with None defaults**

Run: `.venv/Scripts/python.exe -c "from spag4d import SPAG4D; s = SPAG4D(depth_model='da360'); r = s.convert('TestImage/monbachtal_riverbank_primary.jpg', '/tmp/test.ply')"`
Expected: Prints scene defaults, converts successfully with auto-computed parameters.

- [ ] **Step 5: Commit**

```bash
git add spag4d/core.py api.py
git commit -m "feat: wire scene_analysis into convert pipeline, auto defaults"
```

---

### Task 3: Gap-Driven Camera Selection

**Files:**
- Modify: `spag4d-refine/spag4d_refine/camera/trajectory.py` (add `select_gap_cameras`)
- Test: `tests/test_gap_cameras.py`

- [ ] **Step 1: Write test for select_gap_cameras**

```python
# tests/test_gap_cameras.py
import numpy as np
import pytest


def test_select_gap_cameras_returns_requested_count():
    """Should return at most n_cameras cameras."""
    from spag4d_refine.camera.trajectory import select_gap_cameras
    from spag4d_refine.gaussian.cloud import GaussianCloud

    # Create a minimal cloud (10 Gaussians at origin)
    cloud = GaussianCloud(
        means=np.random.randn(10, 3).astype(np.float32) * 0.1,
        scales=np.full((10, 3), 0.01, dtype=np.float32),
        quats=np.tile([0, 0, 0, 1], (10, 1)).astype(np.float32),
        colors=np.random.rand(10, 3).astype(np.float32),
        opacities=np.full((10, 1), 0.9, dtype=np.float32),
    )

    cameras = select_gap_cameras(
        cloud, n_cameras=4, radius=1.0, device="cuda",
    )
    assert len(cameras) <= 4
    assert len(cameras) >= 1  # At least one camera should find gaps


def test_select_gap_cameras_finds_gaps():
    """Cameras should be placed where alpha is low."""
    from spag4d_refine.camera.trajectory import select_gap_cameras
    from spag4d_refine.gaussian.cloud import GaussianCloud

    # Cloud covering only +X hemisphere — -X should have gaps
    means = np.zeros((100, 3), dtype=np.float32)
    means[:, 0] = np.random.uniform(0.5, 2.0, 100)  # All at +X
    means[:, 1] = np.random.uniform(-1, 1, 100)
    means[:, 2] = np.random.uniform(-1, 1, 100)

    cloud = GaussianCloud(
        means=means,
        scales=np.full((100, 3), 0.05, dtype=np.float32),
        quats=np.tile([0, 0, 0, 1], (100, 1)).astype(np.float32),
        colors=np.random.rand(100, 3).astype(np.float32),
        opacities=np.full((100, 1), 0.9, dtype=np.float32),
    )

    cameras = select_gap_cameras(
        cloud, n_cameras=2, radius=3.0, device="cuda",
    )
    # Should have found cameras — the -X side has no coverage
    assert len(cameras) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gap_cameras.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_gap_cameras'`

- [ ] **Step 3: Implement select_gap_cameras**

Add to end of `spag4d-refine/spag4d_refine/camera/trajectory.py`:

```python
def select_gap_cameras(
    cloud: "GaussianCloud",
    n_cameras: int = 4,
    radius: float = 0.5,
    n_candidates: int = 14,
    vfov_deg: float = 60.0,
    resolution: tuple[int, int] = (512, 288),
    alpha_threshold: float = 0.95,
    device: str = "cuda",
) -> CameraSet:
    """
    Select cameras that see the most gaps in the Gaussian splat.

    Renders the splat from candidate viewpoints distributed around the
    scene, ranks them by gap coverage, and returns the top N.

    Args:
        cloud: GaussianCloud to analyze
        n_cameras: Maximum cameras to return
        radius: Distance from scene center
        n_candidates: Number of candidate viewpoints to test
        vfov_deg: Vertical field of view
        resolution: Render resolution (width, height) — low for speed
        alpha_threshold: Coverage above this = no gaps
        device: Torch device

    Returns:
        CameraSet with the cameras that see the most gaps
    """
    import logging
    logger = logging.getLogger(__name__)

    # Generate candidate cameras: icosahedral-ish distribution
    # Use Fibonacci sphere sampling for uniform distribution
    candidates = []
    golden_ratio = (1 + 5 ** 0.5) / 2
    for i in range(n_candidates):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n_candidates)

        # Spherical to Cartesian (Y-up)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.cos(phi)
        z = radius * np.sin(phi) * np.sin(theta)

        eye = np.array([x, y, z])
        cam = PinholeCamera.look_at(
            eye=eye,
            target=np.zeros(3),
            up=np.array([0.0, 1.0, 0.0]),
            vfov_deg=vfov_deg,
            width=resolution[0],
            height=resolution[1],
        )
        candidates.append(cam)

    # Render each candidate and measure gap coverage
    from ..renderer.gsplat_renderer import GsplatRenderer
    renderer = GsplatRenderer(cloud, device=device)

    gap_scores = []
    for i, cam in enumerate(candidates):
        result = renderer.render(cam)
        gap_fraction = float((result.alpha < alpha_threshold).mean())
        gap_scores.append((i, gap_fraction, cam))

    # Sort by gap fraction descending (most gaps first)
    gap_scores.sort(key=lambda x: x[1], reverse=True)

    # Filter out cameras with no meaningful gaps
    min_gap = 0.01  # At least 1% gaps
    viable = [(i, score, cam) for i, score, cam in gap_scores if score >= min_gap]

    if not viable:
        # No gaps found — return a default orbit
        logger.info(f"  No gaps found in {n_candidates} candidates, using default orbit")
        return generate_orbit_trajectory(
            center=np.zeros(3), radius=radius,
            n_cameras=n_cameras, vfov_deg=vfov_deg,
            resolution=resolution,
        )

    # Merge nearby candidates (within 30 degrees)
    selected = []
    for i, score, cam in viable:
        if len(selected) >= n_cameras:
            break
        # Check angular distance to already-selected cameras
        pos = cam.c2w[:3, 3]
        too_close = False
        for _, _, sel_cam in selected:
            sel_pos = sel_cam.c2w[:3, 3]
            cos_angle = np.dot(pos, sel_pos) / (
                np.linalg.norm(pos) * np.linalg.norm(sel_pos) + 1e-8
            )
            if cos_angle > np.cos(np.radians(30)):
                too_close = True
                break
        if not too_close:
            selected.append((i, score, cam))

    # If merging was too aggressive, fill with top remaining
    if len(selected) < n_cameras:
        for i, score, cam in viable:
            if len(selected) >= n_cameras:
                break
            if not any(s[0] == i for s in selected):
                selected.append((i, score, cam))

    cams = [cam for _, _, cam in selected]
    for i, (idx, score, cam) in enumerate(selected):
        logger.info(f"  Camera {i+1}: {score:.1%} gaps (candidate #{idx})")

    return CameraSet(cams)
```

Also add the required import at the top of the file:

```python
from .pinhole import PinholeCamera, CameraSet
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gap_cameras.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d-refine/spag4d_refine/camera/trajectory.py tests/test_gap_cameras.py
git commit -m "feat: gap-driven camera selection via Fibonacci sphere sampling"
```

---

### Task 4: Wire Gap Cameras into Pipeline and UI

**Files:**
- Modify: `spag4d-refine/spag4d_refine/pipeline.py:83-94` (camera generation)
- Modify: `static/index.html` (add "Auto" to camera mode dropdown)
- Modify: `static/js/app.js` (pass "auto" preset)

- [ ] **Step 1: Update pipeline.py to handle "auto" camera preset**

Replace the Stage 1 camera block (lines 83-94) in `pipeline.py`:

```python
        # === Stage 1: Camera trajectory ===
        logger.info("Stage 1: Camera trajectory")
        if cameras is None:
            if self.config.camera_preset == "auto":
                from .camera.trajectory import select_gap_cameras
                cameras = select_gap_cameras(
                    cloud,
                    n_cameras=self.config.n_cameras,
                    radius=self.config.orbit_radius,
                    vfov_deg=self.config.camera_vfov_deg,
                    resolution=(512, 288),
                    device=self.config.device,
                )
            else:
                cameras = generate_preset_trajectory(
                    preset=self.config.camera_preset,
                    center=np.zeros(3),
                    radius=self.config.orbit_radius,
                    n_cameras=self.config.n_cameras,
                    vfov_deg=self.config.camera_vfov_deg,
                    resolution=self.config.render_resolution,
                )
        logger.info(f"  Using {len(cameras)} cameras")
```

- [ ] **Step 2: Update config.py default preset to "auto"**

In `spag4d-refine/spag4d_refine/config.py`, change:

```python
    camera_preset: str = "auto"
```

- [ ] **Step 3: Add "Auto" to UI camera mode dropdown**

In `static/index.html`, find the camera mode select element and add Auto as the first and selected option:

```html
<option "Auto (gap-driven)" (selected) value="auto">
```

- [ ] **Step 4: Test via the single-camera test script**

Run: `.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'spag4d-refine')
from spag4d_refine.config import RefineConfig
c = RefineConfig()
assert c.camera_preset == 'auto'
print('Config default: auto')
"`
Expected: Prints "Config default: auto"

- [ ] **Step 5: Commit**

```bash
git add spag4d-refine/spag4d_refine/pipeline.py spag4d-refine/spag4d_refine/config.py static/index.html static/js/app.js
git commit -m "feat: wire gap-driven cameras into pipeline, Auto mode default"
```

---

### Task 5: Multi-View Color Consistency in Shadow Validator

**Files:**
- Modify: `spag4d-refine/spag4d_refine/seeding/shadow_validator.py`
- Modify: `spag4d-refine/spag4d_refine/config.py` (add threshold)
- Test: `tests/test_color_consistency.py`

- [ ] **Step 1: Write test for color consistency**

```python
# tests/test_color_consistency.py
import numpy as np
import pytest


def test_consistent_colors_promoted():
    """Gaussians seen with consistent colors across views should be promoted."""
    from spag4d_refine.seeding.shadow_validator import validate_shadow_gaussians
    from spag4d_refine.gaussian.cloud import GaussianCloud
    from spag4d_refine.gaussian.provenance import GaussianSource
    from spag4d_refine.camera.pinhole import PinholeCamera

    # One seeded Gaussian at [0, 0, -2] (in front of cameras)
    cloud = GaussianCloud(
        means=np.array([[0.0, 0.0, -2.0]], dtype=np.float32),
        scales=np.full((1, 3), 0.1, dtype=np.float32),
        quats=np.array([[0, 0, 0, 1]], dtype=np.float32),
        colors=np.array([[0.5, 0.3, 0.2]], dtype=np.float32),
        opacities=np.full((1, 1), 0.85, dtype=np.float32),
        provenance=np.array([GaussianSource.SEEDED], dtype=np.int32),
    )

    # Two cameras looking at the Gaussian
    cam1 = PinholeCamera.look_at(
        eye=np.array([0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )
    cam2 = PinholeCamera.look_at(
        eye=np.array([-0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )

    # Both synthesized images have similar color at the Gaussian's projection
    synth1 = np.full((64, 64, 3), 0.5, dtype=np.float32)
    synth2 = np.full((64, 64, 3), 0.52, dtype=np.float32)  # Slightly different

    result = validate_shadow_gaussians(
        cloud, [cam1, cam2],
        synthesized_images=[synth1, synth2],
        color_consistency_threshold=0.15,
        consistency_threshold=0.3,
    )

    assert result.provenance[0] == GaussianSource.PROMOTED
    assert result.opacities[0] > 0.5  # Should keep high opacity


def test_inconsistent_colors_reduced_opacity():
    """Gaussians with conflicting colors across views get reduced opacity."""
    from spag4d_refine.seeding.shadow_validator import validate_shadow_gaussians
    from spag4d_refine.gaussian.cloud import GaussianCloud
    from spag4d_refine.gaussian.provenance import GaussianSource
    from spag4d_refine.camera.pinhole import PinholeCamera

    cloud = GaussianCloud(
        means=np.array([[0.0, 0.0, -2.0]], dtype=np.float32),
        scales=np.full((1, 3), 0.1, dtype=np.float32),
        quats=np.array([[0, 0, 0, 1]], dtype=np.float32),
        colors=np.array([[0.5, 0.3, 0.2]], dtype=np.float32),
        opacities=np.full((1, 1), 0.85, dtype=np.float32),
        provenance=np.array([GaussianSource.SEEDED], dtype=np.int32),
    )

    cam1 = PinholeCamera.look_at(
        eye=np.array([0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )
    cam2 = PinholeCamera.look_at(
        eye=np.array([-0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )

    # Wildly different colors — hallucination
    synth1 = np.full((64, 64, 3), 0.9, dtype=np.float32)  # Bright
    synth2 = np.full((64, 64, 3), 0.1, dtype=np.float32)  # Dark

    result = validate_shadow_gaussians(
        cloud, [cam1, cam2],
        synthesized_images=[synth1, synth2],
        color_consistency_threshold=0.15,
        consistency_threshold=0.3,
    )

    # Should still be promoted (geometrically visible) but with reduced opacity
    assert result.opacities[0] < 0.5, "Opacity should be reduced for color-inconsistent Gaussian"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_color_consistency.py -v`
Expected: FAIL with `TypeError: validate_shadow_gaussians() got an unexpected keyword argument 'synthesized_images'`

- [ ] **Step 3: Add color_consistency_threshold to config**

In `spag4d-refine/spag4d_refine/config.py`, add after `min_seed_confidence`:

```python
    color_consistency_threshold: float = 0.15
    hallucination_opacity: float = 0.3
```

- [ ] **Step 4: Extend validate_shadow_gaussians with color consistency**

Rewrite `spag4d-refine/spag4d_refine/seeding/shadow_validator.py`:

```python
"""Shadow Gaussian validation: provisional → promoted lifecycle."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource
from ..camera.pinhole import PinholeCamera

logger = logging.getLogger(__name__)


def validate_shadow_gaussians(
    cloud: GaussianCloud,
    cameras: List[PinholeCamera],
    consistency_threshold: float = 0.8,
    synthesized_images: Optional[List[np.ndarray]] = None,
    color_consistency_threshold: float = 0.15,
    hallucination_opacity: float = 0.3,
) -> GaussianCloud:
    """
    Multi-view consistency check for shadow Gaussians.

    Projects each SEEDED Gaussian into multiple cameras to check visibility.
    Optionally checks color consistency across synthesized images to detect
    hallucinations.

    Args:
        cloud: GaussianCloud with SEEDED Gaussians
        cameras: Validation cameras
        consistency_threshold: Fraction of cameras for geometric promotion
        synthesized_images: Optional list of [H, W, 3] Klein outputs per camera.
            When provided, color consistency is checked for promoted Gaussians.
        color_consistency_threshold: Max L1 color distance between views (0.15 default)
        hallucination_opacity: Reduced opacity for color-inconsistent Gaussians

    Returns:
        Updated GaussianCloud with SEEDED → PROMOTED or PRUNED
    """
    seeded_mask = cloud.provenance == GaussianSource.SEEDED
    n_seeded = int(np.sum(seeded_mask))

    if n_seeded == 0:
        return cloud

    seeded_means = cloud.means[seeded_mask]
    n_cams = len(cameras)

    if n_cams == 0:
        logger.warning("No validation cameras, promoting all seeded Gaussians")
        new_prov = cloud.provenance.copy()
        new_prov[seeded_mask] = GaussianSource.PROMOTED
        cloud.provenance = new_prov
        return cloud

    # For each seeded Gaussian, count visibility and optionally sample colors
    visibility_counts = np.zeros(n_seeded, dtype=np.int32)
    check_colors = (synthesized_images is not None and len(synthesized_images) == n_cams)

    # Collect per-camera colors for each Gaussian: [n_seeded, n_cams, 3]
    if check_colors:
        sampled_colors = np.full((n_seeded, n_cams, 3), np.nan, dtype=np.float32)

    for cam_idx, cam in enumerate(cameras):
        w2c = cam.w2c
        pts_cam = (w2c[:3, :3] @ seeded_means.T + w2c[:3, 3:4]).T

        # OpenGL convention: camera looks along -Z
        z = pts_cam[:, 2]
        in_front = z < -0.01

        # Project to pixel coords
        neg_z = np.clip(-z, 0.01, None)
        u = pts_cam[:, 0] / neg_z * cam.fx + cam.cx
        v = cam.cy - pts_cam[:, 1] / neg_z * cam.fy

        in_bounds = (
            in_front
            & (u >= 0) & (u < cam.width)
            & (v >= 0) & (v < cam.height)
        )

        visibility_counts += in_bounds.astype(np.int32)

        # Sample synthesized colors at projected positions
        if check_colors:
            visible_idx = np.where(in_bounds)[0]
            if len(visible_idx) > 0:
                px = np.clip(u[visible_idx].astype(int), 0, cam.width - 1)
                py = np.clip(v[visible_idx].astype(int), 0, cam.height - 1)
                synth = synthesized_images[cam_idx]
                sampled_colors[visible_idx, cam_idx] = synth[py, px]

    # Promote if visible in enough cameras
    visibility_ratio = visibility_counts / max(n_cams, 1)
    promote = visibility_ratio >= consistency_threshold

    new_prov = cloud.provenance.copy()
    new_opacities = cloud.opacities.copy()
    seeded_indices = np.where(seeded_mask)[0]
    new_prov[seeded_indices[promote]] = GaussianSource.PROMOTED
    new_prov[seeded_indices[~promote]] = GaussianSource.PRUNED

    # Color consistency check for promoted Gaussians
    n_color_reduced = 0
    if check_colors:
        for local_idx in np.where(promote)[0]:
            colors = sampled_colors[local_idx]  # [n_cams, 3]
            valid_colors = colors[~np.isnan(colors[:, 0])]  # Only cameras that saw it
            if len(valid_colors) < 2:
                continue
            # Max pairwise L1 distance
            max_dist = 0.0
            for i in range(len(valid_colors)):
                for j in range(i + 1, len(valid_colors)):
                    dist = np.abs(valid_colors[i] - valid_colors[j]).mean()
                    max_dist = max(max_dist, dist)
            if max_dist > color_consistency_threshold:
                global_idx = seeded_indices[local_idx]
                new_opacities[global_idx] = hallucination_opacity
                n_color_reduced += 1

    n_promoted = int(np.sum(promote))
    n_pruned = n_seeded - n_promoted
    logger.info(
        f"Shadow validation: {n_seeded:,} seeded → "
        f"{n_promoted:,} promoted, {n_pruned:,} pruned "
        f"(threshold={consistency_threshold:.0%}, {n_cams} cameras)"
    )
    if check_colors and n_color_reduced > 0:
        logger.info(
            f"  Color consistency: {n_color_reduced:,} Gaussians had opacity "
            f"reduced to {hallucination_opacity} (threshold={color_consistency_threshold})"
        )

    cloud.provenance = new_prov
    cloud.opacities = new_opacities
    return cloud
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_color_consistency.py -v`
Expected: Both tests PASS

- [ ] **Step 6: Commit**

```bash
git add spag4d-refine/spag4d_refine/seeding/shadow_validator.py spag4d-refine/spag4d_refine/config.py tests/test_color_consistency.py
git commit -m "feat: multi-view color consistency check in shadow validator"
```

---

### Task 6: Wire Color Consistency into Pipeline

**Files:**
- Modify: `spag4d-refine/spag4d_refine/pipeline.py` (pass synth_targets to validator)

- [ ] **Step 1: Pass synthesized images and config thresholds to validate_shadow_gaussians**

In `pipeline.py`, find the Stage 7 shadow validation call and update it:

```python
            # === Stage 7: Shadow validation ===
            _progress(self.session.round_number, "Shadow validation", 90)
            from .seeding.shadow_validator import validate_shadow_gaussians
            cloud = validate_shadow_gaussians(
                cloud, list(cameras),
                consistency_threshold=self.config.promotion_consistency_threshold,
                synthesized_images=synth_targets if synth_targets else None,
                color_consistency_threshold=getattr(self.config, 'color_consistency_threshold', 0.15),
                hallucination_opacity=getattr(self.config, 'hallucination_opacity', 0.3),
            )
```

- [ ] **Step 2: Test the integration**

Run the single-camera test script to verify the pipeline still works end-to-end with color consistency enabled. With a single camera, color consistency won't trigger (needs 2+ views), but it should not crash.

- [ ] **Step 3: Commit**

```bash
git add spag4d-refine/spag4d_refine/pipeline.py
git commit -m "feat: wire color consistency into refinement pipeline"
```

---

### Task 7: Final Integration Test and Push

- [ ] **Step 1: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py tests/test_gap_cameras.py tests/test_color_consistency.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run the single-camera refinement test end-to-end**

Verify the full pipeline works with new auto defaults by running `test_single_camera_refine.py` and inspecting output images.

- [ ] **Step 3: Push to GitHub**

```bash
git push
```
