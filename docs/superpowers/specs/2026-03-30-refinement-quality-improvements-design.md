# Refinement Pipeline Quality Improvements

**Date:** 2026-03-30
**Status:** Approved
**Goal:** Make the Klein refinement pipeline robust across indoor and outdoor scenes without manual tuning.

---

## 1. Scale-Relative Parameters

Replace hardcoded meter values with depth-distribution-relative thresholds so the pipeline auto-adapts to any scene scale.

### Parameter Mapping

| Parameter | Current Default | Proposed Default |
|-----------|----------------|-----------------|
| Sky cutoff | 80m | 95th percentile of depth |
| Orbit radius | 0.5m | 5% of median scene depth |
| Depth range min | 0.1m | 1st percentile of depth |
| Depth range max | 100m | 99th percentile of depth |
| Confidence decay | 50px fixed | 2% of image height |

### Implementation

- New function `compute_scene_defaults(depth_map: np.ndarray) -> dict` in `spag4d/scene_analysis.py`.
- Called from `core.py` immediately after depth estimation, before SPAG conversion.
- Parameters passed as `"auto"` from the API/UI resolve to concrete meter values via this function.
- The UI displays the resolved values so the user can see and override them.
- Edge cutoff (65 degrees) is already scale-invariant and stays as-is.

### Files

- `spag4d/scene_analysis.py` (new)
- `spag4d/core.py` (call site after depth estimation)
- `api.py` (pass "auto" defaults)
- `static/index.html` (show resolved values)

---

## 2. Gap-Driven Camera Placement

Analyze the splat to find where gaps actually are, then place cameras only at viewpoints that see disocclusions. Reduces Klein inference from N fixed cameras to only the cameras that matter.

### Process

1. After SPAG conversion, render the splat from 12-16 candidate viewpoints (icosahedral sampling at auto-computed radius from Section 1).
2. Render at low resolution (512x288) for speed (~1s total for all candidates).
3. For each candidate, compute alpha coverage. Skip viewpoints with >95% coverage (no gaps).
4. Rank remaining candidates by TYPE_C gap area (pixel count).
5. Select top N cameras where N = user's camera count setting (default 4).
6. Optionally merge nearby candidates that see the same gap (angular distance < 30 degrees).

### UI Integration

- New "Auto" option in the Camera Mode dropdown, set as default.
- User still sets camera count — the pipeline picks the best N instead of evenly spacing them.
- Diagnostics gallery shows which viewpoints were selected and their gap percentages.

### Files

- `spag4d-refine/spag4d_refine/camera/trajectory.py` — new `select_gap_cameras(cloud, n_cameras, radius, resolution=(512, 288)) -> CameraSet`
- `spag4d-refine/spag4d_refine/pipeline.py` — call `select_gap_cameras` when preset is "auto"
- `static/index.html` — add "Auto" to camera mode dropdown
- `static/js/app.js` — pass "auto" preset

### Dependencies

- Requires Section 1 for auto-computed orbit radius.
- Uses existing `GsplatRenderer` for candidate renders.
- Uses existing `classify_frame` for gap detection.

---

## 3. Multi-View Color Consistency

Extend the shadow validator to detect hallucinated colors by comparing synthesized RGB across cameras. Catches per-camera Klein hallucinations without adding new model dependencies.

### Process

1. For each SEEDED Gaussian visible in 2+ cameras, project it to pixel coordinates in each camera (already done in shadow validator).
2. Sample the Klein synthesized RGB at that pixel from each camera's output.
3. Compute pairwise L1 color distance between all camera pairs.
4. If max color disagreement > 0.15 (configurable), mark the Gaussian as unreliable.
5. Unreliable Gaussians get opacity reduced to 0.3 instead of being promoted at full confidence. This lets the original splat show through rather than leaving a hole.

### Interface Change

`validate_shadow_gaussians()` gains an optional `synthesized_images: List[np.ndarray]` parameter. When provided, color consistency is checked alongside geometric visibility. When omitted, behavior is unchanged (backward compatible).

### Files

- `spag4d-refine/spag4d_refine/seeding/shadow_validator.py` — add color sampling and consistency check
- `spag4d-refine/spag4d_refine/config.py` — add `color_consistency_threshold: float = 0.15`
- `spag4d-refine/spag4d_refine/pipeline.py` — pass `synth_targets` to validator

---

## Implementation Order

1. **Scale-relative parameters** — unblocks auto radius for Section 2
2. **Gap-driven camera placement** — depends on Section 1
3. **Multi-view color consistency** — independent, can be built in parallel with Section 2

## Out of Scope

- SSIM loss or densification during optimization (future work)
- Faster model alternatives (existing backend fallback chain is sufficient)
- Track 2 / InSpatio-World integration
- Changes to Klein/LoRA inference
