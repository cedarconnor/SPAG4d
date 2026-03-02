# SPAG-4D v2 Design Document

## Vision

A focused tool that converts 360 panoramic images into 3D Gaussian Splats using Apple's ML-SHARP model, with DAP metric depth for global scale alignment. Simple web UI for upload, conversion, and viewing.

---

## Table of Contents

1. [What We Learned from Reference Projects](#1-what-we-learned-from-reference-projects)
2. [Root Cause of Current Quality Issues](#2-root-cause-of-current-quality-issues)
3. [Architecture Overview](#3-architecture-overview)
4. [Pipeline Specification](#4-pipeline-specification)
5. [File-by-File Plan](#5-file-by-file-plan)
6. [Web UI Specification](#6-web-ui-specification)
7. [What Gets Removed](#7-what-gets-removed)
8. [Fresh Start vs Modify](#8-fresh-start-vs-modify)
9. [Implementation Phases](#9-implementation-phases)

---

## 1. What We Learned from Reference Projects

### ml-sharp-pinokio and sharp-gui

Both are single-perspective-image wrappers around SHARP. Neither handles panoramas. But they produce **color-accurate output** because they use Apple's pipeline correctly:

| Step | Reference Projects | Our Pipeline (Current) |
|------|-------------------|----------------------|
| SHARP inference | `predictor(image, disparity_factor)` | Same |
| NDC -> world | Apple's `unproject_gaussians()` | Same (SHARP does this internally) |
| Color space | Apple's `save_ply()` converts **linearRGB -> sRGB** before SH0 encoding | We store **raw linearRGB** in SH0 — **WRONG** |
| SH0 encoding | `(sRGB - 0.5) / 0.28209479` | `(linearRGB - 0.5) / 0.28209479` — encoding linear values as if they were sRGB |
| Opacity encoding | Logit: `log(o / (1-o))` | Logit — same, correct |
| Scale encoding | Log: `log(scale)` | Log — same, correct |
| Quaternion order | WXYZ in PLY (`rot_0`=W) | WXYZ in PLY — same, correct |
| Viewer | Standard renderers expecting sRGB SH0 | Custom WebGL viewer + SuperSplat |

**The critical bug**: Standard 3DGS renderers (SuperSplat, GaussianSplats3D, gsplat) decode SH0 as: `color = SH0 * 0.28209479 + 0.5`, then render directly to screen assuming sRGB. If we encode linearRGB values in SH0, the renderer interprets them as sRGB, producing incorrect brightness and desaturated colors.

Apple's `save_ply` comment says it explicitly:
> *"Public renderers do not have linearRGB->sRGB conversions after rendering. If they render linearRGB Gaussians as-is, the output would be dark without Gamma correction. To make it compatible to public renderers, we force convert linearRGB to sRGB during export."*

### What the reference projects DON'T need to solve (but we do)

Since reference projects handle single perspective images, they skip these panorama-specific challenges:

1. **Multi-face merging** — stitching 6 or 20 SHARP runs into a coherent 360 scene
2. **Global depth alignment** — SHARP's per-face depth has arbitrary scale; DAP provides metric reference
3. **Seam artifacts** — different SHARP runs produce different Gaussians at face boundaries
4. **Pole handling** — cubemap distortion at top/bottom requires care
5. **Scale consistency** — adjacent faces may produce Gaussians at slightly different scales

---

## 2. Root Cause of Current Quality Issues

### Issue 1: Color encoding (PRIMARY)

Our PLY writer encodes SHARP's linearRGB colors directly into SH0 coefficients. When a standard renderer decodes them, it treats linearRGB as sRGB, producing wrong colors.

**Fix**: Apply `linearRGB2sRGB()` before SH0 encoding, exactly as Apple's `save_ply()` does.

### Issue 2: Too much custom machinery

Our pipeline manually unpacks Gaussians, does custom coordinate transforms, applies DAP alignment, jitter, ownership masks, overlap corrections, and seam smoothing. Each step is a place where bugs can hide. The reference projects succeed by staying close to Apple's code.

**Fix**: Simplify. Use Apple's `save_ply()` as the reference for encoding. Minimize custom transforms.

### Issue 3: Accumulated experimental code

PanDA, DA3, video processing, adaptive stride, visual odometry, depth-pro fusion — all add complexity without contributing to the 360 SHARP workflow.

**Fix**: Remove everything that isn't DAP + SHARP + web UI.

---

## 3. Architecture Overview

```
Input: 360 ERP panorama image
  |
  v
[DAP Depth Model] --> metric depth map [H, W]
  |
  v
[Projector] --> N perspective face crops + per-face rotation matrices
  |            (Cubemap: 6 faces, 90 FOV, 0.4 overlap)
  |            (Icosahedral: 20 faces, 70 FOV, 0.3 overlap)
  v
[SHARP Model] --> per-face Gaussians3D (NDC -> camera-space, done internally)
  |
  v
[Per-Face Processing]
  1. DAP scale alignment (log-median radial distance matching)
  2. Rotate to world frame (face rotation matrix)
  3. Nearest-face ownership filter (hard Voronoi assignment)
  4. Depth range filter
  v
[Global Post-Processing]
  1. Overlap scale corrections (cross-face depth consistency)
  2. Seam color smoothing (cross-face color blending)
  3. Outlier pruning (statistical, optional)
  v
[Export]
  PLY: linearRGB -> sRGB -> SH0, logit opacity, log scale, WXYZ quaternions
  |
  v
[Web Viewer: GaussianSplats3D]
  Standard .ply loading, orbit controls, progressive rendering
```

### Core modules (7 files)

| Module | Responsibility |
|--------|---------------|
| `core.py` | Orchestrator: loads models, calls pipeline, returns result |
| `sharp_gaussian_pipeline.py` | Per-face SHARP inference + merge + post-processing |
| `projection.py` | CubemapProjector / IcosahedralProjector |
| `dap_model.py` | DAP depth estimation wrapper |
| `ply_writer.py` | Standard 3DGS PLY export (with correct sRGB encoding) |
| `scene_filter.py` | Sky detection, outlier pruning |
| `cli.py` | CLI interface (convert, serve, download-models) |

### Supporting modules (3 files)

| Module | Responsibility |
|--------|---------------|
| `spherical_grid.py` | Spherical geometry helpers, quaternion math |
| `depth_blend.py` | Depth map blending (used by DAP face compositing) |
| `depth_refiner.py` | Edge-guided depth refinement |

---

## 4. Pipeline Specification

### 4.1 Input

- 360 equirectangular panorama (JPEG, PNG, WebP)
- Aspect ratio ~2:1 (standard ERP)
- Resolution: any (faces are extracted at `cubemap_size`, default 1536)

### 4.2 DAP Depth Estimation

- Run DAP on the full ERP image
- Output: metric depth map `[H, W]` in meters (radial distance, not planar Z)
- Used ONLY for global scale alignment — not for Gaussian geometry

### 4.3 Face Extraction

**Cubemap mode** (default, 6 faces):
- FOV: 90 degrees
- Overlap: 40% (total FOV per face = 126 degrees)
- Face size: 1536x1536 (SHARP's expected resolution)
- Directions: +X, -X, +Y, -Y, +Z, -Z

**Icosahedral mode** (20 faces):
- FOV: 70 degrees
- Overlap: 30%
- Face size: 1536x1536
- Directions: 20 icosahedral vertices

### 4.4 Per-Face SHARP Processing

For each face:

1. **Resize to 1536x1536** if needed (SHARP's fixed internal resolution)
2. **Compute disparity_factor**: `f_px / face_size` where `f_px = face_size / (2 * tan(fov/2))`
3. **Run SHARP forward pass**: produces `Gaussians3D` with ~1.18M points (768x768 x 2 layers)
4. **Unpack dual layers**: keep all front-layer Gaussians; keep back-layer where `opacity > 0.3`
5. **DAP scale alignment**: project Gaussian positions to ERP coords, bilinearly sample DAP depth, compute `scale_factor = exp(median(log(dap/sharp_radial)))`, multiply means and scales by this factor
6. **Apply face rotation**: `means_world = R @ means_cam`, `quats_world = q_face * q_cam`
7. **Ownership filter**: assign each Gaussian to closest face by angular distance; discard if not owned by current face
8. **Depth filter**: discard Gaussians outside `[depth_min, depth_max]`

### 4.5 Global Post-Processing

1. **Concatenate** all face results
2. **Overlap scale corrections**: compare radial distances in face-boundary overlap zones; compute per-face multiplicative correction via binned log-ratio median matching
3. **Seam color smoothing**: blend colors near Voronoi boundaries toward cross-face neighbor averages (strength 0.3, margin 0.15)
4. **Outlier pruning** (optional): statistical outlier removal via cKDTree nearest-neighbor distance

### 4.6 Export

**PLY format** (standard 3DGS, compatible with all viewers):

```
Per-vertex properties (14 x float32 = 56 bytes):
  x, y, z                    -- world-space position
  f_dc_0, f_dc_1, f_dc_2    -- SH degree-0 coefficients (sRGB-encoded)
  opacity                    -- logit: log(o / (1-o))
  scale_0, scale_1, scale_2  -- log(singular_value)
  rot_0, rot_1, rot_2, rot_3 -- quaternion WXYZ
```

**Color encoding** (matching Apple's save_ply):
```python
srgb = linearRGB2sRGB(colors)           # gamma correction
sh0 = (srgb - 0.5) / sqrt(1 / (4*pi))  # SH0 encoding
```

### 4.7 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `projection_mode` | `"cubemap"` | cubemap, icosahedral | Face layout |
| `cubemap_size` | `1536` | 512-2048 | Face extraction resolution |
| `depth_min` | `0.1` | 0.01-10 | Near clip (meters) |
| `depth_max` | `100.0` | 10-1000 | Far clip (meters) |
| `sky_threshold` | `80.0` | 10-500 | Sky removal distance |
| `grid_jitter` | `0.0` | 0.0-0.5 | Sub-pixel jitter to break grid pattern |
| `outlier_pruning` | `0.0` | 0.0-1.0 | Statistical outlier removal strength |
| `seam_smooth` | `0.3` | 0.0-1.0 | Seam color blending strength |

---

## 5. File-by-File Plan

### 5.1 Files to MODIFY

#### `spag4d/ply_writer.py` — Fix color encoding

**Current**: Encodes raw colors (linearRGB) into SH0.
**Change**: Add `linearRGB2sRGB()` conversion before SH0 encoding.

```python
# BEFORE (wrong):
f_dc = (color - 0.5) / SH_C0

# AFTER (correct, matching Apple's save_ply):
srgb = linearRGB_to_sRGB(color)
f_dc = (srgb - 0.5) / SH_C0
```

Add the standard sRGB gamma function:
```python
def _linearRGB_to_sRGB(linear):
    """IEC 61966-2-1 standard, matching Apple Metal Spec 7.7.7"""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.clip(linear, 0.0031308, None) ** (1.0 / 2.4) - 0.055
    )
```

#### `spag4d/sharp_gaussian_pipeline.py` — Clean up

- Remove the dead `_linear_to_srgb` function (already done)
- Keep the pipeline logic as-is (it's correct for geometry)
- The color fix happens at the export layer (ply_writer), not here

#### `spag4d/core.py` — Simplify

- Remove PanDA and DA3 code paths
- Remove video-related parameters
- Remove `sharp_refiner` hybrid path (the "refine 2D maps onto ERP grid" approach)
- Keep: DAP depth + SHARP direct pipeline + outlier pruning

#### `spag4d/cli.py` — Simplify

- Remove `convert-video` command
- Remove PanDA/DA3 depth model options
- Keep: `convert`, `serve`, `download-models`

#### `api.py` — Simplify

- Remove video endpoints (`/api/convert_video`, `/api/preview_video`, `/api/download_video`)
- Remove PanDA/DA3 model initialization
- Remove video-related parameters from `/api/convert`
- Keep: convert, status, preview, download, health, shutdown
- Add CORS headers for GaussianSplats3D SharedArrayBuffer:
  ```python
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```

#### `static/index.html` — Rebuild UI

- Remove: video tab, PanDA/DA3 model selectors, DA3 projection dropdown
- Replace custom WebGL viewer with GaussianSplats3D
- Keep: file upload, conversion controls, progress display, download button
- Add: Load .ply from disk button (for testing external files)

#### `static/js/app.js` — Rebuild

- Remove: video mode, video preview, PanDA/DA3 UI logic
- Integrate GaussianSplats3D viewer via import map
- Serve converted .ply file to viewer via URL
- Support file upload from disk (Blob URL)

### 5.2 Files to REMOVE

| File | Reason |
|------|--------|
| `spag4d/panda_model.py` | PanDA depth model — not needed |
| `spag4d/da3_model.py` | Depth Anything V3 — not needed |
| `spag4d/visual_odometry.py` | Video stabilization — not needed |
| `spag4d/adaptive_stride.py` | Experimental, never wired in |
| `spag4d/gaussian_params.py` | NumPy reference impl, unused |
| `spag4d/sharp_refiner.py` | Hybrid 2D-map refinement approach — replaced by direct pipeline |
| `spag4d/sharp_depth_fusion.py` | SHARP monodepth fusion — DAP handles depth |
| `spag4d/depth_pro_fusion.py` | Legacy DepthPro class (keep helpers if needed by projection) |
| `spag4d/splat_writer.py` | Custom binary format — switching to standard PLY + GaussianSplats3D |
| `static/js/splat-viewer.js` | Custom WebGL viewer — replaced by GaussianSplats3D |
| `static/js/pano-viewer.js` | Three.js pano preview — can use GaussianSplats3D or drop |
| `test_single_face.py` | Diagnostic script — served its purpose |
| `test_hot_swap.py` | Test script |
| `test_panda_depth.py` | PanDA test script |
| `src/ml-depth-pro/` | Unused Apple Depth Pro clone |

### 5.3 Files to ADD

| File | Purpose |
|------|---------|
| `static/js/viewer.js` | Thin wrapper around GaussianSplats3D — init, load scene, handle upload |

### 5.4 Dependencies

**Keep**:
- `torch`, `torchvision` — ML models
- `numpy`, `Pillow` — image processing
- `plyfile` — PLY read/write
- `scipy` — cKDTree, signal processing
- `opencv-python` — guided filter
- `fastapi`, `uvicorn`, `python-multipart` — web server
- `huggingface_hub` — model downloads
- `click` — CLI
- `tqdm` — progress bars
- `einops`, `safetensors`, `omegaconf` — DAP architecture
- `sharp` (Apple ML-SHARP) — core model

**Add**:
- `@mkkellogg/gaussian-splats-3d` (CDN, no pip) — web viewer
- `three` (CDN, v0.164+) — GaussianSplats3D dependency

**Remove**:
- `imageio-ffmpeg` — video only
- `hatchling` — DA3 build only

---

## 6. Web UI Specification

### Layout

```
+----------------------------------------------------------+
|  SPAG-4D                              [Load PLY] [Help]  |
+----------------------------------------------------------+
|                                                          |
|                   GaussianSplats3D                       |
|                   Viewer Canvas                          |
|                   (orbit, pan, zoom, WASD)               |
|                                                          |
+----------------------------------------------------------+
|  Upload: [Choose File]  [Convert]                        |
|                                                          |
|  Projection: [Cubemap v]    Face Size: [1536]            |
|  Depth Range: [0.1] - [100]                              |
|  Sky Threshold: [80]                                     |
|  Grid Jitter: [0.0 ----o---- 0.5]                        |
|  Seam Smoothing: [0.3 ----o---- 1.0]                     |
|  Outlier Pruning: [0.0 ----o---- 1.0]                    |
|                                                          |
|  Progress: [=============>          ] 65%                |
|  Status: Processing face 4/6...                          |
|                                                          |
|  [Download PLY]                                          |
+----------------------------------------------------------+
```

### Viewer Integration

```html
<script type="importmap">
{
    "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.164.0/build/three.module.js",
        "@mkkellogg/gaussian-splats-3d": "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
    }
}
</script>
```

### Loading converted results

After conversion completes, the server has the PLY at `/api/download/{job_id}?format=ply`. The viewer loads it:

```javascript
viewer.addSplatScene(`/api/download/${jobId}?format=ply`, {
    showLoadingUI: true,
    splatAlphaRemovalThreshold: 5
}).then(() => viewer.start());
```

### Loading from disk

User clicks "Load PLY", selects a file, we create a Blob URL:

```javascript
const blob = new Blob([arrayBuffer]);
const url = URL.createObjectURL(blob);
viewer.addSplatScene(url, { showLoadingUI: true });
```

### Required server headers

FastAPI middleware for SharedArrayBuffer support:

```python
@app.middleware("http")
async def add_coop_coep(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response
```

---

## 7. What Gets Removed

### By category

| Category | Items Removed | Lines Saved |
|----------|--------------|-------------|
| Depth models | PanDA, DA3, DepthPro fusion | ~980 |
| Video pipeline | visual_odometry, video endpoints, video UI | ~700 |
| Experimental | adaptive_stride, gaussian_params | ~510 |
| Hybrid path | sharp_refiner, sharp_depth_fusion | ~740 |
| Custom viewer | splat_writer, splat-viewer.js, pano-viewer.js | ~1,700 |
| Test files | test_*.py, src/ml-depth-pro | ~600 |
| **Total** | | **~5,230 lines** |

### What remains

| Component | Lines (approx) |
|-----------|----------------|
| core.py (simplified) | ~350 |
| sharp_gaussian_pipeline.py | ~600 |
| projection.py | ~400 |
| dap_model.py | ~340 |
| ply_writer.py (fixed) | ~220 |
| scene_filter.py | ~430 |
| spherical_grid.py | ~200 |
| depth_blend.py | ~400 |
| depth_refiner.py | ~200 |
| cli.py (simplified) | ~250 |
| api.py (simplified) | ~500 |
| index.html (simplified) | ~250 |
| app.js (simplified) | ~300 |
| viewer.js (new) | ~100 |
| **Total** | **~4,540 lines** |

This is roughly a **50% reduction** from the current ~9,500 lines.

---

## 8. Fresh Start vs Modify

### Recommendation: **Modify the existing project**

| Factor | Fresh Start | Modify Existing |
|--------|------------|-----------------|
| Working DAP integration | Must rewrite | Already works |
| Working projectors (cube + icosa) | Must rewrite | Already works |
| Working SHARP inference | Must rewrite | Already works |
| Working API/job system | Must rewrite | Keep + simplify |
| Working face merging + seam logic | Must rewrite | Already works |
| Color bug fix | Same effort | Same effort |
| Dead code removal | Not needed | ~2 hours of deletion |
| Risk of losing working code | High | Low |
| Time estimate | 3-5x longer | 1x baseline |
| Git history | Lost | Preserved |

**The core pipeline is architecturally sound.** The face extraction, SHARP inference, DAP alignment, face-to-world rotation, ownership filtering, and seam smoothing are all correct approaches. The problems are:

1. **Color encoding bug** — 3-line fix in `ply_writer.py`
2. **Accumulated dead code** — deletion, not rewriting
3. **UI complexity** — swap viewer library, remove unused controls

None of these require rearchitecting the pipeline. A fresh start would mean reimplementing ~4,500 lines of working geometry/projection/blending code just to get back to where we are, minus the dead weight.

### The plan

1. Fix the color encoding (30 minutes)
2. Delete dead modules and code paths (1-2 hours)
3. Swap the viewer to GaussianSplats3D (2-3 hours)
4. Simplify the UI (1-2 hours)
5. Test end-to-end (ongoing)

---

## 9. Implementation Phases

### Phase 1: Fix Color Accuracy (CRITICAL)

**Goal**: Get color-correct output from the existing pipeline.

1. Add `linearRGB_to_sRGB()` to `ply_writer.py`
2. Apply it in `save_ply_gsplat()` before SH0 encoding
3. Test: convert a panorama, load PLY in SuperSplat — colors should match source
4. This single fix may resolve the majority of the visual quality gap vs reference projects

### Phase 2: Remove Dead Code

**Goal**: Halve the codebase.

1. Delete removed modules (panda, da3, visual_odometry, adaptive_stride, gaussian_params, sharp_refiner, sharp_depth_fusion, depth_pro_fusion, splat_writer)
2. Strip PanDA/DA3/video code from core.py, cli.py, api.py
3. Delete test files and src/ml-depth-pro
4. Update requirements.txt and pyproject.toml
5. Verify: `python -m spag4d serve` still starts, conversion still works

### Phase 3: Swap Viewer

**Goal**: Replace custom WebGL viewer with battle-tested GaussianSplats3D.

1. Add import map for Three.js + GaussianSplats3D (CDN)
2. Create `static/js/viewer.js` — thin wrapper (init, loadScene, loadFromFile)
3. Update `static/index.html` — replace canvas with viewer container
4. Update `static/js/app.js` — wire viewer to conversion results and file upload
5. Add COOP/COEP headers to FastAPI for SharedArrayBuffer
6. Delete `static/js/splat-viewer.js` and `static/js/pano-viewer.js`
7. Test: upload panorama, convert, verify viewer shows result with orbit controls

### Phase 4: Simplify UI

**Goal**: Clean, minimal interface.

1. Remove video tab and controls
2. Remove PanDA/DA3 model selector
3. Streamline parameter controls to essential settings only
4. Clean up CSS
5. Test: full end-to-end workflow in browser

### Phase 5: Quality Tuning (Ongoing)

**Goal**: Maximize output quality now that the foundation is correct.

1. Compare output against reference projects (same input image, crop to perspective, compare colors)
2. Tune seam smoothing parameters
3. Evaluate icosahedral mode vs cubemap for seam visibility
4. Investigate whether grid jitter helps or hurts at various levels
5. Consider SH degree 1 support for view-dependent effects (future)
