# SPAG-4D Quality Architecture — Build Plan

Based on the design doc [SPAG4D_Quality_Architecture.md](file:///d:/SPAG-4D/SPAG4D_Quality_Architecture.md). Phases are reordered for build efficiency:
- **Apple Depth Pro (Phase 1) is deferred to last**, as requested.
- Phases 2 & 3 are first because they have no new model dependencies.
- Phases 4 & 5 refine the output and depend on Phase 3's sky mask.

> [!IMPORTANT]
> This plan touches [gaussian_converter.py](file:///d:/SPAG-4D/spag4d/gaussian_converter.py), [ply_writer.py](file:///d:/SPAG-4D/spag4d/ply_writer.py), [splat_writer.py](file:///d:/SPAG-4D/spag4d/splat_writer.py), and [core.py](file:///d:/SPAG-4D/spag4d/core.py). Each phase is independently deployable and backward-compatible.

---

## Phase Order

| Sprint | Phase | Name | Why First |
|--------|-------|------|-----------|
| 1 | **Phase 2** | Depth-Aware Gaussian Sizing + Normal-Oriented Covariances | Biggest visual win, zero new models |
| 2 | **Phase 3** | Sky Detection & Pole Thinning | Removes worst artifacts, CPU-only |
| 3 | **Phase 5** | Adaptive Stride | Depends on Phase 3 sky mask |
| 4 | **Phase 4** | Feathered Blending / Laplacian Fusion | Refines depths, standalone |
| 5 | **Phase 1** | Cubemap Depth Pro Fusion | New model (Apple Depth Pro), deferred last |

Phase 6 (Tangent Patch Upgrade) is explicitly out-of-scope for this sprint.

---

## Proposed Changes

### Sprint 1 — Phase 2: Gaussian Parameterization

---

#### [NEW] [gaussian_params.py](file:///d:/SPAG-4D/spag4d/gaussian_params.py)

New module with all per-Gaussian parameterization logic extracted from the design doc:

- `compute_gaussian_scales(depth_map, stride, H, W, base_scale_factor)` → depth-proportional [(scale_iso, scale_h, scale_v)](file:///d:/SPAG-4D/spag4d/da3_model.py#53-91)
- `estimate_normals_from_erp_depth(depth_map, H, W)` → [(normals [H,W,3], confidence [H,W])](file:///d:/SPAG-4D/spag4d/da3_model.py#53-91)
- `normal_to_covariance(normal, scale_h, scale_v, thickness_ratio)` → [(quat_3dgs [4], log_scales [3])](file:///d:/SPAG-4D/spag4d/da3_model.py#53-91)
- `generate_gaussians(erp_image, depth_map, positions, normals, normal_conf, ...)` → list of Gaussian dicts with full anisotropic parameterization

#### [MODIFY] [ply_writer.py](file:///d:/SPAG-4D/spag4d/ply_writer.py)

Add `rot_0..3` (quaternion w,x,y,z) and `scale_0..2` (log-scale xyz) property fields. Currently writes uniform isotropic scales; update to accept per-Gaussian quaternion + anisotropic scales from `gaussian_params.py`.

#### [MODIFY] [splat_writer.py](file:///d:/SPAG-4D/spag4d/splat_writer.py)

Same field additions as [ply_writer.py](file:///d:/SPAG-4D/spag4d/ply_writer.py) for `.splat` binary format.

#### [MODIFY] [gaussian_converter.py](file:///d:/SPAG-4D/spag4d/gaussian_converter.py)

Replace the current uniform isotropic Gaussian generation loop with calls to the new `gaussian_params` module. The conversion function gains two new optional parameters: `--oriented-gaussians` (default: on) and `--thickness`.

#### [MODIFY] [core.py](file:///d:/SPAG-4D/spag4d/core.py)

Wire the new `generate_gaussians()` into the main pipeline, passing normals and normal confidence through.

---

### Sprint 2 — Phase 3: Sky Detection & Pole Thinning

---

#### [NEW] [scene_filter.py](file:///d:/SPAG-4D/spag4d/scene_filter.py)

- `detect_sky_gradient(depth_map, erp_image, depth_max)` — fast, no-model sky detection
- `detect_sky_depth(depth_map, depth_max, threshold_ratio)` — simple depth threshold method
- `compute_pole_thinning_mask(H, W, stride, min_density_ratio)` — sin(θ)-based keep mask
- `get_adaptive_stride_per_row(H, base_stride, max_stride_factor)` — latitude-proportional stride
- `filter_gaussian_candidates(depth_map, erp_image, stride, sky_mode, pole_thinning)` → [(keep_mask, sky_mask)](file:///d:/SPAG-4D/spag4d/da3_model.py#53-91)

`SkyMode` enum: `SKIP`, `BACKGROUND_SPHERE`, `LOW_OPACITY`.

#### [MODIFY] [gaussian_converter.py](file:///d:/SPAG-4D/spag4d/gaussian_converter.py)

Call `filter_gaussian_candidates()` before generating Gaussians. Add `--sky-mode`, `--sky-radius`, and `--no-pole-thinning` CLI parameters.

#### [MODIFY] [core.py](file:///d:/SPAG-4D/spag4d/core.py)

Pass `sky_mode` and `pole_thinning` through to the main convert pipeline.

#### [MODIFY] [api.py](file:///d:/SPAG-4D/api.py)

Expose `sky_mode` and `pole_thinning` in the REST API request model.

---

### Sprint 3 — Phase 5: Adaptive Stride

---

#### [NEW] [adaptive_stride.py](file:///d:/SPAG-4D/spag4d/adaptive_stride.py)

- `compute_adaptive_stride_map(depth_map, base_stride, min_stride, max_stride, depth_reference)` → integer stride map
- `refine_stride_at_edges(stride_map, depth_map, edge_reduction_factor)` — Canny edge detection, dilate, reduce stride in edge zones
- `sample_with_adaptive_stride(depth_map, erp_image, normals, stride_map, sky_mask, pole_mask)` → list of [(row, col)](file:///d:/SPAG-4D/spag4d/da3_model.py#53-91) positions
- `compute_stride_for_budget(depth_map, target_count, sky_mask, ...)` → base stride that achieves the target Gaussian count

#### [MODIFY] [gaussian_converter.py](file:///d:/SPAG-4D/spag4d/gaussian_converter.py)

Replace uniform `[::stride, ::stride]` sampling with `sample_with_adaptive_stride()` when `--adaptive-stride` is set. Add `--target-gaussians` and `--edge-refine` CLI options.

---

### Sprint 4 — Phase 4: Laplacian / Poisson Depth Blending

---

#### [NEW] [depth_blend.py](file:///d:/SPAG-4D/spag4d/depth_blend.py)

- `build_gaussian_pyramid(image, n_levels)` 
- `build_laplacian_pyramid(gaussian_pyramid)`
- `reconstruct_from_laplacian(laplacian_pyramid)`
- `laplacian_depth_fusion(dap_depth, dp_depth, n_levels, low_freq_cutoff)` — primary fusion method
- `masked_laplacian_fusion(dap_depth, dp_depth, dp_confidence, n_levels)` — confidence-weighted
- `poisson_blend_faces(face_depths, face_uv_maps, target_shape)` — advanced gradient-domain blend
- `DepthBlender` class wrapping all modes

`BlendMode` enum: `FEATHERED`, `LAPLACIAN`, `POISSON`.

#### [MODIFY] [core.py](file:///d:/SPAG-4D/spag4d/core.py)

After any depth estimation (DAP or future Depth Pro), pass through `DepthBlender.fuse_dap_and_depth_pro()`. Add `--blend-mode` and `--blend-levels` parameters.

---

### Sprint 5 (Last) — Phase 1: Cubemap Depth Pro Fusion

---

> [!NOTE]
> Requires `apple/ml-depth-pro` installed (already handled by [install.bat](file:///d:/SPAG-4D/install.bat) via `ml-sharp` clone pattern). Add a parallel clone step for `ml-depth-pro` to [install.bat](file:///d:/SPAG-4D/install.bat).

#### [NEW] [depth_pro_fusion.py](file:///d:/SPAG-4D/spag4d/depth_pro_fusion.py)

- `DepthProFusion` class:
  - [load_model()](file:///d:/SPAG-4D/spag4d/sharp_refiner.py#76-107) — lazy-loads Depth Pro from `ml-depth-pro`
  - `fuse(erp_image, dap_depth)` → [(fused_depth [H,W], confidence [H,W])](file:///d:/SPAG-4D/spag4d/da3_model.py#53-91)
  - `_project_to_cubemap(erp_image)` — 6× 100° FOV face projection
  - `_run_depth_pro(faces)` — per-face inference
  - `_align_to_dap(face_depths, dap_depth)` — scale/shift alignment using robust median
  - `_composite(aligned_faces, dap_depth)` — feathered edge-distance blending back to ERP

`ProjectionMode` enum: `CUBEMAP` (6 faces, fast), `ICOSAHEDRON` (20 faces, higher quality).

#### [MODIFY] [install.bat](file:///d:/SPAG-4D/install.bat)

Add a clone + install step for `apple/ml-depth-pro` into `src/ml-depth-pro`, similar to the existing DA3 and SHARP steps.

#### [MODIFY] [core.py](file:///d:/SPAG-4D/spag4d/core.py)

Insert the `DepthProFusion.fuse()` call between DAP depth and SHARP refinement when `--depth-pro-fuse` is set. Pass `fused_depth` and `confidence` to `DepthBlender` for Laplacian fusion.

#### [MODIFY] [api.py](file:///d:/SPAG-4D/api.py)

Expose `depth_pro_fuse`, `projection`, and `face_size` in the REST API.

---

## Verification Plan

### Existing Tests

- [tests/test_coordinates.py](file:///d:/SPAG-4D/tests/test_coordinates.py) — spherical coordinate math; will catch regressions in normal estimation
- [tests/test_da3_integration.py](file:///d:/SPAG-4D/tests/test_da3_integration.py) — DA3 depth model integration
- [tests/test_ply_compat.py](file:///d:/SPAG-4D/tests/test_ply_compat.py) — PLY file format validity; **must pass after Phase 2 adds quaternion + scale fields**

Run existing tests:
```
cd d:\SPAG-4D
python_embed\python.exe -m pytest tests/ -v
```

### New Tests to Add

#### `tests/test_gaussian_params.py` (Phase 2)

- `test_scale_proportional_to_depth()` — scales at depth=10 should be ~2× scales at depth=5
- `test_normal_points_inward()` — `dot(normal, xyz_position) < 0` for >95% of Gaussians
- `test_covariance_flat_disc()` — log_scales[2] (normal axis) should be `thickness_ratio * min(s0, s1)`
- `test_ply_has_quaternion_fields()` — PLY output contains `rot_0`, `scale_0` etc.

Run with: `python_embed\python.exe -m pytest tests/test_gaussian_params.py -v`

#### `tests/test_scene_filter.py` (Phase 3)

- `test_no_sky_gaussians_in_output()` — with `sky_mode=SKIP`, no Gaussian `xyz` should have radius ≥ `depth_max * 0.95`
- `test_pole_density_uniform()` — after pole thinning, CV of density per steradian band < 0.5
- `test_sky_mask_has_sky_pixels()` — sky detection returns non-empty mask on outdoor image

Run with: `python_embed\python.exe -m pytest tests/test_scene_filter.py -v`

#### `tests/test_adaptive_stride.py` (Phase 5)

- `test_budget_within_10_percent()` — `compute_stride_for_budget(target=500_000)` produces count within 10%
- `test_stride_increases_with_depth()` — deeper pixels have higher stride values
- `test_edge_stride_reduced()` — stride at Canny edges is lower than surrounding non-edge pixels

Run with: `python_embed\python.exe -m pytest tests/test_adaptive_stride.py -v`

### Manual Verification

After each sprint, do a quick visual smoke test using the test panorama:

1. Start the server: `double-click run.bat`
2. Open browser to `http://localhost:7860`
3. Load [test_panorama_8k.png](file:///d:/SPAG-4D/test_panorama_8k.png) and Convert with default settings
4. Open the output [.ply](file:///d:/SPAG-4D/test_panda_tiles.ply) in [SuperSplat](https://supersplat.online) or [gsplat viewer](https://antimatter15.com/splat/)
5. Check visually:
   - **Phase 2**: Splats near camera are small/crisp; far splats are large and fill gaps
   - **Phase 3**: Sky region is empty (or shows sky sphere); poles have no blob
   - **Phase 5**: Dense detail at edges; sparse in uniform backgrounds
   - **Phase 4**: No visible seams or depth discontinuities at blending boundaries
   - **Phase 1 (last)**: Sharper object edges compared to DAP-only output
