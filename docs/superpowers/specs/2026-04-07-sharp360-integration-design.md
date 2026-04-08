# SHARP 360 Integration Design

**Goal:** Add Apple's SHARP model as a third splat generator alongside DA360 and DAP, with native Windows SeedVR2 image upscaling and DA360-based inter-face alignment.

**Architecture:** The UI/CLI generator selector dispatches to one of three paths: DA360 (depth + SPAG projection), DAP (depth + SPAG projection), or SHARP 360 (per-face ML prediction + alignment + merge). All three produce standard PLY files. Refinement (GSFix3D, OmniRoam) works on any PLY regardless of source generator.

---

## 1. Generator Selection

The top-level concept changes from "depth model" to "generator":

| Generator | Pipeline | Output |
|-----------|----------|--------|
| **DA360** (default) | DA360 depth estimation -> SPAG spherical projection | Standard PLY |
| **DAP** | DAP metric depth -> SPAG spherical projection | Standard PLY |
| **SHARP 360** | N perspective faces -> optional SeedVR2 upscale -> SHARP prediction per face -> DA360 alignment -> rotate + merge | Standard PLY |

The `depth_model` parameter remains as a backward-compatible alias for `generator=da360` and `generator=dap`.

---

## 2. SHARP 360 Pipeline

New module: `spag4d/sharp360.py` (~300-400 lines).

### Pipeline steps:

1. **Load panorama** -- validate 2:1 equirectangular format.
2. **Build extraction layout** -- compute N horizon views with overlap.
   - Default: 6 sides, ~70 deg FOV each with 10 deg overlap.
   - Each view defined by a `FaceOrientation` (right/down/forward rotation matrix).
   - Face size derived from `panorama_width / side_count` (e.g. 8192/6 = 1365 px).
   - Focal length derived from FOV and face size.
3. **Extract perspective faces** -- ERP to perspective projection per view using bilinear sampling with horizontal wrap at the 360 boundary.
4. **Optional SeedVR2 upscale** -- native Windows image upscale of each face before SHARP inference. Focal lengths adjusted proportionally to the upscale ratio.
5. **SHARP prediction** -- load Apple checkpoint (~500 MB, auto-downloaded on first use), run `predict_image()` per face. Each face produces a `Gaussians3D` (positions, scales, quaternions, colors, opacities) directly from the image.
6. **Border clipping** -- hard Voronoi: clip each face's Gaussians to its angular sector (360/N degrees per side). No soft/feathered blending (causes ghosting, same lesson as our existing face merging).
7. **DA360 alignment** -- run DA360 on the full panorama to get a disparity map. Extract per-view disparity slices. Align SHARP depths to DA360 using a smooth NxN grid scale field (grid_resolution=8, detail_weight=0.0). This corrects inter-face scale disagreement without destroying SHARP's fine geometric detail.
8. **Rotate + merge** -- rotate each face's Gaussians into world frame using the view's 3x4 affine transform, then concatenate all faces.
9. **Global scale restore** -- restore median scene radius to the pre-alignment value so the scene doesn't shrink/grow.
10. **PLY export** -- save using SHARP's own `save_ply()` from `sharp.utils.gaussians`, which handles linearRGB to sRGB conversion and SH0 encoding correctly for SHARP's color space. This is the same function the SHARP_360_to_Splat repo uses.

### Key functions extracted from SHARP_360_to_Splat repo:

These are extracted and adapted (not wrapped) into `sharp360.py`:

- `extract_perspective_view()` -- ERP to perspective bilinear projection
- `build_extraction_layout()` / `make_horizon_view()` -- view geometry computation
- `filter_gaussians_by_view_border()` -- hard Voronoi Gaussian clipping
- `align_gaussians_to_reference()` -- DA360 grid-based depth alignment
- `scale_gaussians()` / `apply_per_point_scales()` -- Gaussian scaling helpers
- `merge_gaussians()` -- concatenate Gaussians3D across faces
- `bilinear_sample()` / `bilinear_sample_scalar()` -- sampling utilities

### Vendored dependency:

`ml-sharp/` vendored under `spag4d/sharp_arch/ml-sharp/` following the same pattern as `spag4d/da360_arch/DA360/`. Key types used:

- `sharp.models.PredictorParams`, `sharp.models.create_predictor` -- model loading
- `sharp.cli.predict.predict_image` -- per-face inference
- `sharp.utils.gaussians.Gaussians3D` -- Gaussian data structure
- `sharp.utils.gaussians.apply_transform` -- affine rotation
- `sharp.utils.gaussians.save_ply` -- PLY export (linearRGB to sRGB conversion)
- `sharp.utils.color_space.linearRGB2sRGB` -- gamma correction

---

## 3. Native SeedVR2 Adapter

New module: `spag4d/seedvr2.py` (~150 lines). Replaces the WSL2 adapter entirely.

### Two modes:

**Image mode** (for SHARP face upscaling):
- Input: dict of `{name: np.ndarray}` face images + temp directory
- Writes faces as PNGs to input dir
- Calls `inference_cli.py` via `subprocess.run(sys.executable, ...)` (native Python)
- Reads back upscaled PNGs from output dir
- Returns: upscaled face dict, new width, new height

**Video mode** (for OmniRoam pipeline Stage 3):
- Input: video file path
- Calls `inference_cli.py` with `--video_backend opencv`
- Returns: path to upscaled video

### Shared settings:
- Model: `seedvr2_ema_3b_fp16.safetensors` (default, ~6.4 GB)
- Color correction: `lab`
- Block swap: configurable (default 0 for native, no offload needed)
- Resolution targeting: configurable target short-side pixels
- `--dit_offload_device cpu` when block swap > 0

### Installation:
- SeedVR2 code vendored at `third_party/seedvr2_videoupscaler/`
- Model weights stored in `third_party/seedvr2_videoupscaler/models/`
- `python -m spag4d download-models --model seedvr2` downloads weights

---

## 4. Integration Points

### `core.py`:
- `SPAG4D.__init__()` accepts `generator` parameter (default `"da360"`)
- `convert()` dispatches: `da360`/`dap` -> existing SPAG path, `sharp360` -> new SHARP path
- Both paths return `ConversionResult` with identical shape

### `api.py`:
- `/api/convert` accepts `generator=da360|dap|sharp360` (backward compatible: `depth_model` still works for da360/dap)
- SHARP-specific params: `side_count` (default 6), `seedvr2_upscale` (default false)
- Response shape unchanged

### `cli.py`:
- `python -m spag4d convert panorama.jpg output.ply --generator sharp360`
- `--side-count 6`, `--seedvr2-upscale` flags for SHARP path
- `--depth-model da360` remains as alias for `--generator da360`

### UI (`index.html`, `app.js`):
- Top dropdown label changes from "Depth Model" to "Generator" with options: DA360, DAP, SHARP 360
- When SHARP 360 selected: show side count dropdown (4/6/8/10/12), SeedVR2 upscale checkbox
- When DA360/DAP selected: show existing stride/depth/sky controls (unchanged)
- Refinement panel unchanged

### `pipeline_v2.py` Stage 3:
- Rewired from WSL2 `run_seedvr2_upscale()` to native `seedvr2.upscale_video()`
- Same parameters, different subprocess mechanism (native Python instead of WSL2 bash)

### Refinement defaults per generator:
- DA360/DAP: current defaults (hole_mask_threshold=0.3, convergence_threshold=0.02)
- SHARP 360: stored in config, potentially different hole detection threshold since SHARP seams produce different hole patterns than depth-discontinuity holes

---

## 5. Files Changed

### Removed:
- `spag4d/refine/seedvr2_adapter.py` -- WSL2 SeedVR2 adapter
- WSL2 SeedVR2 path conversion logic from `omniroam_adapter.py`
- SeedVR2 installation section from `scripts/setup_omniroam_wsl.sh`
- `tests/test_seedvr2_adapter.py` -- replaced by tests for native adapter

### New:
- `spag4d/sharp360.py` -- SHARP 360 pipeline orchestrator
- `spag4d/seedvr2.py` -- native Windows SeedVR2 adapter (image + video)
- `spag4d/sharp_arch/ml-sharp/` -- vendored Apple SHARP model code
- `tests/test_sharp360.py` -- SHARP pipeline unit tests
- `tests/test_seedvr2_native.py` -- native SeedVR2 adapter tests

### Modified:
- `spag4d/core.py` -- generator dispatch
- `spag4d/cli.py` -- `--generator`, `--side-count`, `--seedvr2-upscale` args
- `api.py` -- generator param on convert endpoint
- `static/index.html` -- generator dropdown, SHARP settings panel
- `static/js/app.js` -- generator routing, conditional UI
- `spag4d/refine/pipeline_v2.py` -- Stage 3 uses native SeedVR2
- `spag4d/refine/omniroam_config.py` -- SeedVR2 settings updated for native path
- `spag4d/refine/omniroam_adapter.py` -- remove SeedVR2 WSL2 logic

### Unchanged:
- `spag4d/refine/pipeline.py` -- GSFix3D path
- `spag4d/spag_converter.py` -- SPAG projection
- `spag4d/da360_model.py` -- DA360 model (reused by SHARP alignment)
- `spag4d/dap_model.py` -- DAP model
- `spag4d/ply_writer.py` -- PLY export
- All other refine modules

---

## 6. Model Weights

| Model | Size | Source | Download |
|-------|------|--------|----------|
| SHARP checkpoint | ~500 MB | `ml-site.cdn-apple.com` | Auto on first use via `torch.hub` |
| DA360 Large | ~1.3 GB | Existing | Already cached |
| SeedVR2 3B | ~6.4 GB | HuggingFace | `python -m spag4d download-models --model seedvr2` |

---

## 7. VRAM Requirements

| Generator | Conversion VRAM | With SeedVR2 Upscale |
|-----------|----------------|---------------------|
| DA360 | ~2 GB | N/A |
| DAP | ~3 GB | N/A |
| SHARP 360 | ~6 GB (SHARP) + ~2 GB (DA360 alignment) | +~8 GB for SeedVR2 (sequential, not concurrent) |

SHARP and DA360 run sequentially (DA360 is freed before SHARP loads, or vice versa) to keep peak VRAM manageable.

---

## 8. License

- SHARP model code: Apple MIT-equivalent license (see `ml-sharp/LICENSE`)
- SHARP model weights: Apple model license (noncommercial research, see `ml-sharp/LICENSE_MODEL`)
- SeedVR2: MIT
- DA360: existing license unchanged
- Core SPAG-4D: remains MIT. SHARP integration is an optional module.
