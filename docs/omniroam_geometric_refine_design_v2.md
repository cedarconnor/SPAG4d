# SPAG-4D Design Document: Geometric OmniRoam Refine

**Version**: 2.0
**Module**: `spag4d.refine.geometric`
**Status**: Proposed
**Replaces/supplements**: `spag4d.refine.v2` (OmniRoam photometric distillation)
**Target hardware**: Single A6000 (48 GB VRAM), Windows native + WSL2 for OmniRoam
**Scope**: Parallax disocclusion repair for DA360/DAP/SHARP 360–generated panoramic gsplats

**Changes from v1**: Added cross-frame consistency gate (§5.6) as the primary safeguard against single-frame inventions. Pinned depth convention and introduced the `is_nearer_than_rendered` helper (§5.3.6). Extended validity mask with sky and specular exclusion (§5.3.2). Deferred ICP (§5.4) and base-splat provenance filter (§5.5.3) to v1.1. Reworked aggregated-gaussian provenance to be aggregation-aware (§5.8). Made color polish SH degree configurable (§5.9). Added geometry-stability metrics to the validation plan (§10.3). Relabeled G5 as a target pending benchmark.

---

## 1. Motivation

SPAG-4D's current OmniRoam refine path (`refine_splat_v2`) generates an 81-frame ERP walkthrough video, extracts perspective crops that overlap hole regions, and distills those crops into the base splat as pseudo-supervision via L1+SSIM photometric loss at `tier2_weight=0.20`. This works but leaves substantial signal unused and introduces failure modes that the current weight-tuning heuristic can only partially mitigate:

1. **Perspective cropping discards panoramic coverage.** Each 81-frame ERP video is reduced to a small number of crop windows, throwing away most of what OmniRoam's holistic representation provides.
2. **Photometric-only supervision is weak and risky.** OmniRoam is a generative model. Its frames contain plausible but fabricated content in hole regions, and photometric loss bakes those fabrications directly into gaussian SH coefficients, scale, and (through densification) position. The `tier2_weight=0.20` band-aid trades fill quality for drift risk and never fully solves either.
3. **No geometric signal flows from OmniRoam back into the splat.** The video's implicit 3D structure — which is exactly what parallax disocclusion repair needs — is never extracted.
4. **Statistical mismatch.** Even when photometric distillation succeeds, new gaussians spawned via densification have different scale distributions and provenance than DA360-generated ones, which is visually perceptible along fill boundaries.

The holes SPAG-4D actually needs to fill are, empirically, almost entirely **parallax disocclusions**: surfaces the input panorama could have seen geometrically if nearer content weren't occluding them from the capture viewpoint. The foreground silhouette heavily constrains what the occluded region can be, and OmniRoam's temporal coherence plus a spherical depth backend are together sufficient to recover that geometry without generative photometric loss at all.

## 2. Goals and Non-Goals

### Goals

- **G1.** Fill parallax disocclusion holes in DA360/DAP/SHARP 360–generated splats using OmniRoam video as input, via geometric fusion rather than photometric distillation.
- **G2.** Reuse the existing depth generator backends (`da360`, `dap`, `sharp360`) so injected gaussians inherit the same statistical fingerprint (scale distribution, depth-noise profile, radial-to-Z convention, provenance metadata) as the base splat.
- **G3.** Never degrade well-observed regions of the base splat. The pipeline must be strictly additive on regions where the base splat already has confident coverage.
- **G4.** Fit within the existing `refine_splat_v2(...)`-style entry point and `OmniRoamConfig`-style configuration surface so users can swap backends without restructuring their scripts.
- **G5 (target, pending benchmark).** Aim for end-to-end wall-clock time within 1.5× of `refine_splat_v2` on a single A6000. This is an aspiration, not a commitment: Stage 1 (per-frame depth estimation) and the repeated base-splat renders in Stage 2 are the dominant new costs and have not been measured against the v2 path. Runtime will be reported in M9 and the target revisited if missed by more than 2×.
- **G6.** Emit structured diagnostics per OmniRoam frame so failure modes are observable without re-running the pipeline.

### Non-Goals

- **NG1.** Hallucination of truly unobserved regions (surfaces that no view — real or generated — ever observes). That's a score-distillation problem and belongs in a separate track.
- **NG2.** Dynamic scene handling. Static scenes only.
- **NG3.** Replacement of `refine_splat_v2`. The geometric refine will ship alongside v2 as an alternate backend; v2 remains available for users who prefer photometric distillation.
- **NG4.** Training new models. Every component uses existing checkpoints already in SPAG-4D's ecosystem.

## 3. Approach Overview

```
Input:  base_splat.ply  (from da360 / dap / sharp360)
        panorama.jpg
        base_depth_map  (the depth used to build base_splat)

Stage 0: OmniRoam generation     (unchanged from v2 — reuse existing)
         panorama -> N ERP frames + trajectory poses P_i
         (optional SeedVR2 upscale)

Stage 1: Per-frame depth estimation
         For each ERP frame i:  d_i = depth_generator(frame_i)
         Reuses DA360 / SHARP 360 backend — same function that built the base splat
         Also computes sky_mask_i and specular_mask_i for use in Stage 2.

Stage 2: Per-frame depth alignment
         For each frame i:
           render base splat from P_i  ->  rgb_i, depth_rendered_i, alpha_i
           build validity mask M_i (alpha + depth + gradient + ¬sky + ¬specular)
           solve (s_i, t_i) robustly   ->  s_i * d_i + t_i ≈ depth_rendered_i  on M_i
           d_i_aligned = s_i * d_i + t_i
           convert to camera-forward z-depth (one-time, normative convention)
           unproject d_i_aligned using same spherical math as base generator
           -> candidate_points_i  (in base splat world frame)

Stage 3: Per-frame hole filtering
         For each candidate point:
           look up rendered alpha along its ray
           look up rendered depth along its ray
           primary:   alpha < ALPHA_HOLE              -> keep ("alpha")
           secondary: alpha in [ALPHA_HOLE, ALPHA_CONFIDENT]
                      AND is_nearer_than_rendered(...) -> keep ("disoccl")
           else: discard

Stage 3.5: Cross-frame mutual support gate    [NEW IN v2]
         Concatenate all per-frame survivors. KDTree.
         For each point, count distinct supporting frames within radius.
         Drop singletons. Drop disoccl points with < N+1 supporters.
         Emit trajectory_coverage_warning if drop rate exceeds threshold.

Stage 4: Aggregation
         voxel downsample at base-splat local spacing
         carry per-point source metadata for provenance

Stage 5: Gaussian initialization
         each voxel -> one new gaussian
         position = voxel centroid
         scale    = k * local kNN distance (same heuristic as DA360 generator)
         rotation = identity (DA360 convention)
         opacity  = initial 0.35
         SH_0     = averaged color from contributing OmniRoam frames
         provenance = primary_source_frame_idx + support_count + alignment + hole_mode

Stage 6: Color polish (optional, on by default)
         freeze everything except new-gaussian SH coefficients (degree ≤ 1 default)
         render full splat from panorama pose
         L1+SSIM against panorama for ~500 iterations
         only boundary gaussians (visible from panorama) receive gradient

Output: refined_splat.ply  with updated GaussianProvenance metadata
        diagnostics.json   with per-frame alignment, support counts, drop rates,
                           trajectory coverage warnings
        provenance.json    optional sidecar with full per-voxel contributor lists
```

No photometric distillation. No densification. The existing splat is never touched except by additive injection, and the final color polish only updates SH on the new gaussians. ICP and base-splat provenance filtering are deferred to v1.1.

## 4. Module Structure

New module lives under `spag4d/refine/geometric/`. The existing `spag4d/refine/v2/` is untouched.

```
spag4d/
  refine/
    __init__.py              # exports refine_splat_v2 AND refine_splat_geometric
    v2/                      # existing OmniRoam photometric distillation
    geometric/               # NEW
      __init__.py
      pipeline.py            # refine_splat_geometric — orchestration
      config.py              # GeometricRefineConfig dataclass
      depth_align.py         # per-frame scale/shift solve, validity mask
      masks.py               # sky / specular masking helpers
      depth_convention.py    # is_nearer_than_rendered + radial→z helpers
      hole_filter.py         # alpha / disoccl gating
      consistency.py         # cross-frame mutual support gate
      aggregate.py           # voxel downsample
      init_gaussians.py      # PLY injection + provenance
      color_polish.py        # SH-only finetune
      diagnostics.py         # per-frame reports, JSON writer
      render_utils.py        # gsplat depth/alpha render via cube-face composition
```

Dependencies:
- `gsplat` (already used) for differentiable rasterization of base splat from OmniRoam poses
- `open3d` for voxel downsample and KDTree (promote to hard dep for the geometric refine path)
- `scipy` for robust IRLS in `depth_align` and KDTree fallback
- Reuse `spag4d.generators.da360`, `spag4d.generators.sharp360` for Stage 1 — no new depth code

## 5. Detailed Stage Specifications

### 5.1 Stage 0: OmniRoam Generation (unchanged)

Reuse the existing v2 OmniRoam invocation. The only behavioral change is that we now consume the full ERP frames directly rather than perspective crops. Trajectory mode selection, SeedVR2 upscaling, and WSL2 plumbing are unchanged.

Output from Stage 0:
- `frames: list[np.ndarray]` — N ERP frames, shape `(H, W, 3)` at `(1024, 2048)` post-SeedVR2 or `(480, 960)` raw
- `poses: np.ndarray` — shape `(N, 4, 4)`, each row a camera-to-world transform in OmniRoam's trajectory frame
- `frame_scale_hint: float | None` — OmniRoam's reported world scale (usually unreliable; informational only)

### 5.2 Stage 1: Per-Frame Depth Estimation

```python
def estimate_frame_depths(
    frames: list[np.ndarray],
    generator: Literal["da360", "dap", "sharp360"],
) -> list[FrameDepthResult]:
    """
    Run the same depth backend used to build the base splat on each OmniRoam frame.
    Returns affine-invariant depth maps plus sky/specular masks for each frame.
    """
```

`FrameDepthResult` carries:
- `depth_raw`: affine-invariant spherical depth (radial distance, native to the generator)
- `sky_mask`: boolean, True for sky pixels
- `specular_mask`: boolean, True for specular/blown-out pixels

Implementation notes:
- For DA360/DAP, each frame is already equirectangular, so we feed directly to the spherical depth network.
- For SHARP 360, we run per-face depth estimation on the same cube faces SHARP uses for the base splat, then stitch back to ERP using the existing DA360-alignment pass from `sharp360.py`.
- Depth is returned as **affine-invariant** in radial form. Conversion to camera-forward z-depth happens once, after Stage 2's alignment solve, per §5.3.6.
- Batch frames where VRAM permits. DA360 at 1024×2048 on A6000: ~6 frames in a batch. Process 81 frames in ~14 batches, ~2–3 min total.
- Sky and specular masks are computed in the same pass and cached for Stage 2; cost is negligible.

### 5.3 Stage 2: Per-Frame Depth Alignment

This is the most critical stage. Getting it wrong produces floating duplicate geometry and visible seams; getting it right makes everything downstream trivial.

#### 5.3.1 Rendering the base splat

For each frame `i`, render the base splat from pose `P_i` using gsplat's differentiable rasterizer, requesting RGB, depth, and alpha outputs. Since OmniRoam frames are ERP and gsplat is perspective, we render **six cube faces** and compose them into an ERP buffer using the same latlong mapping already present in `spag4d.generators.da360`.

```python
def render_base_from_pose(
    base_model: GaussianModel,
    pose: np.ndarray,                # (4,4) cam-to-world
    resolution: tuple[int, int],     # (H_erp, W_erp)
    near: float = 0.01,
    far: float = 1000.0,
) -> RenderOutput:
    """
    Returns:
      rgb    : (H, W, 3) float32
      depth  : (H, W)    float32, camera-forward z-depth (per §5.3.6)
      alpha  : (H, W)    float32, accumulated opacity in [0, 1]
    """
```

The depth output is **already in camera-forward z**, the normative pipeline convention. Cube-face depths are converted from per-face camera-z to ERP latlong with the radial component projected back to z relative to the ERP camera frame during composition.

#### 5.3.2 Validity mask

Not every pixel of the base-splat render is trustworthy for alignment. We build a mask `M_i` that excludes:

- `alpha_i < ALPHA_CONFIDENT` (default 0.9) — low-coverage regions of the base splat
- `depth_i > FAR_CLAMP_RATIO * far` (default 0.95) — sky / far-plane pixels
- Regions with high depth gradient: `|∇depth_i| / depth_i > GRADIENT_THRESHOLD` (default 0.15) — silhouettes where the splat's depth is most uncertain
- A 3-pixel erosion of the above to avoid boundary bleed
- `sky_mask_i` (from Stage 1): any depth estimate on sky pixels is untrustworthy and destabilizes the scale/shift solve
- `specular_mask_i` (from Stage 1): chrome, water highlights, and blown-out windows that monocular depth estimators consistently hallucinate

```
M_i = alpha_high ∧ depth_finite ∧ low_gradient ∧ ¬sky_mask_i ∧ ¬specular_mask_i ∧ erode(...)
```

If SPAG-4D already has a sky segmenter elsewhere, reuse it. Otherwise the heuristic is HSV-based: high value, low saturation, upper hemisphere of the ERP frame. Specular mask is low local texture variance + high luminance over a 9×9 window. Both are 20-line single-pass image ops; false positives only remove pixels from the alignment solve, they don't corrupt output.

`M_i` typically covers 60–85% of the frame for well-registered OmniRoam poses near the panorama origin, dropping as the trajectory moves away.

#### 5.3.3 Robust scale/shift solve

Over the valid mask, solve for per-frame scale `s_i` and shift `t_i`:

```
minimize_{s, t}  Σ_{p ∈ M_i}  ρ_Huber( s · d_i(p) + t − depth_rendered_i(p) )
```

Implementation via IRLS (iteratively reweighted least squares) with Huber kernel width `δ = 0.05 · median(depth_rendered_i[M_i])`. Converges in 5–10 iterations. Closed-form weighted least squares at each iter. No autograd needed.

```python
@dataclass
class AlignmentResult:
    scale: float
    shift: float
    inlier_count: int
    inlier_fraction: float
    residual_median: float
    residual_p95: float
    converged: bool
```

If `inlier_fraction < 0.2` OR `residual_p95 / median_depth > 0.25`, mark the frame as **alignment failed** and skip it in Stage 3. Log to diagnostics.

#### 5.3.4 Scale-only vs scale-shift

DA360 outputs affine-invariant depth; both scale and shift need solving. DAP outputs scale-invariant depth (some configurations); only scale is solved. `GeometricRefineConfig.align_mode` selects between `"scale_shift"` (default, safe) and `"scale_only"` (tighter, faster convergence when correct).

#### 5.3.5 Unprojection

Once `d_i_aligned = s_i · d_i + t_i`, convert from radial to camera-forward z (§5.3.6), then unproject using the **same** spherical-to-Cartesian math the base generator uses:

```python
from spag4d.generators.da360 import unproject_erp_depth_to_points
points_i = unproject_erp_depth_to_points(
    depth=d_i_aligned_z,
    pose=pose_i,
    radial_to_z=base_model.metadata.radial_to_z,
)
```

This guarantees convention consistency with the base splat down to the floating-point level.

#### 5.3.6 Depth Convention (normative)

All depth values inside the geometric refine pipeline use the following convention. This is normative: every module must conform, and every comparison routes through the helper below.

- **Type**: camera-forward z-depth (distance along the camera optical axis), not radial distance.
- **Sign**: positive forward, zero at the camera, increasing with distance.
- **Units**: base splat world units (meters if the base splat is metric, otherwise the splat's intrinsic unit).
- **Conversion**: DA360/DAP raw output is radial distance to the spherical surface; conversion to camera-forward z-depth happens **once**, immediately after Stage 2's scale/shift solve, using the existing `radial_to_z` helper inherited from the base model metadata.

**"Nearer than" means smaller z.** Always. A candidate point that is "in front of" a rendered surface has `candidate_z < rendered_z`.

All depth comparisons in Stages 3–5 route through this helper:

```python
def is_nearer_than_rendered(
    candidate_z: np.ndarray,      # (K,) camera-forward z, per convention above
    rendered_z: np.ndarray,       # (K,) camera-forward z, per convention above
    margin_ratio: float = 0.02,
    local_depth_scale: np.ndarray | float | None = None,
) -> np.ndarray:                   # (K,) bool
    """
    Returns True for candidates that sit strictly in front of the rendered surface
    by more than margin_ratio * local_depth_scale.

    If local_depth_scale is None, uses rendered_z itself (scale-invariant margin).
    Otherwise, typically pass the local median rendered depth over a neighborhood.
    """
    margin = margin_ratio * (local_depth_scale if local_depth_scale is not None else rendered_z)
    return candidate_z < (rendered_z - margin)
```

Inequalities of the form `a < b - margin` in the pipeline code are banned; all such checks must go through `is_nearer_than_rendered`. Unit test `test_depth_convention.py` verifies that the pipeline fails loudly if any frame-level depth buffer arrives in radial form (detected via a cheap consistency check against rendered gsplat depth).

### 5.4 Stage 2.5: ICP Refinement (deferred to v1.1)

ICP-based pose correction is deferred. The base scale/shift solve plus the cross-frame consistency gate (§5.6) should cover most OmniRoam pose drift. We don't yet know whether residual rotational misregistration matters in practice. Ship without ICP, measure residuals in M9 benchmarks, add ICP in v1.1 only if data shows it matters.

Specification preserved in §11bis for v1.1 work.

### 5.5 Stage 3: Per-Frame Hole Filtering

For each candidate point from frame `i`, determine whether it represents a genuine hole fill. Two signals in v1, both routed through the depth convention helper:

```python
@dataclass
class HoleFilterConfig:
    alpha_hole_threshold: float = 0.30       # primary
    alpha_confident_threshold: float = 0.90  # boundary between primary and secondary
    depth_disoccl_margin_ratio: float = 0.02 # secondary: fraction of local median depth
```

```python
def filter_candidate_points_per_frame(
    candidates: np.ndarray,                  # (K, 3)
    candidate_z: np.ndarray,                 # (K,) camera-forward z in pose_i frame
    source_frame_idx: int,
    rendered_depth: np.ndarray,              # (H, W) camera-forward z
    rendered_alpha: np.ndarray,              # (H, W)
    pose: np.ndarray,
    config: HoleFilterConfig,
) -> FilterResult:
    """
    Returns kept points plus per-point metadata:
      - hole_mode: "alpha" | "disoccl"
      - source_frame_idx: int
      - src_pixel: (u, v) in frame space
    """
```

#### 5.5.1 Primary: Alpha gating

Project each candidate into the rendered alpha buffer. Keep if `alpha < alpha_hole_threshold`. This is the main signal and what makes the approach safe: only regions with no existing confident coverage are ever modified.

#### 5.5.2 Secondary: Depth disocclusion

A candidate point is a parallax disocclusion fill if all hold:

1. Alpha is in the band `[alpha_hole_threshold, alpha_confident_threshold]` (some coverage exists but it's not confident).
2. The candidate projects into a pixel where `is_nearer_than_rendered(candidate_z, rendered_z, margin_ratio=depth_disoccl_margin_ratio, local_depth_scale=local_median_depth)` returns True. `local_median_depth` is computed in a 5×5 window around the projected pixel.
3. The candidate survives the cross-frame consistency gate in §5.6.

#### 5.5.3 Tertiary: Provenance override (deferred to v1.1)

The base splat's `GaussianProvenance` `background_leak` flag would let us treat known-leak regions as holes even at higher alpha. This requires per-gaussian source fidelity the current base generators don't yet provide reliably, and the alpha + disocclusion + consistency triple is strong enough on its own for v1. Specification preserved in §11bis.

#### 5.5.4 Reject

Points that project to a pixel where `alpha >= alpha_confident` AND sit behind the rendered depth are either duplicate geometry or depth-estimation noise. Discard.

Typical kept fraction per frame after the per-frame stage alone: 5–15% of input pixels. The cross-frame gate in §5.6 reduces this further.

### 5.6 Stage 3.5: Cross-Frame Mutual Support

This is the central safeguard of the v2 design. After Stage 3 produces per-frame candidate points but **before** voxel aggregation, reject candidates that lack corroboration from other frames. This prevents single-frame OmniRoam inventions from becoming permanent gaussians, and detects trajectories that didn't give sufficient multi-view coverage.

The motivating concern: a robust scale/shift solve in Stage 2 can fit overlap regions well while still producing inconsistent geometry in hole regions, because hole regions contribute zero residual to the alignment solve. Good residuals on visible geometry do not guarantee trustworthy extrapolation into invisible geometry. The only defense is to require **multiple frames to independently agree** in the hole regions themselves.

```python
@dataclass
class ConsistencyConfig:
    min_support_count: int = 2           # at least N frames must agree
    support_radius: float | None = None  # None -> 1.5 * voxel_size
    positional_std_max: float | None = None  # None -> 0.5 * voxel_size
    strict_mode_for_disoccl: bool = True # require N+1 for secondary-gate points
```

#### 5.6.1 Algorithm

1. Concatenate all kept points from all frames into a single `(K_total, 3)` array with per-point source frame index and hole mode.
2. Build a KDTree over the full array.
3. For each point `p_k`, query all neighbors within `support_radius`. Count the number of **distinct source frames** among neighbors (excluding `p_k` itself).
4. If `distinct_frame_count < min_support_count`, mark as unsupported.
5. For points flagged `hole_mode == "disoccl"` (secondary gate), if `strict_mode_for_disoccl` is True, require `min_support_count + 1` instead. Disocclusion candidates are higher-risk and deserve stricter corroboration.
6. Additionally compute the positional standard deviation of the supporting neighborhood. If it exceeds `positional_std_max`, mark as unsupported (candidates agree on location loosely, but not tightly enough for a single voxel).
7. Drop all unsupported points.

#### 5.6.2 Edge cases

- Points near the trajectory endpoints legitimately have fewer supporting frames. If diagnostic data shows the unsupported rate is concentrated at trajectory endpoints, log a warning but do not auto-relax the threshold — this is a trajectory coverage issue, not a filter issue.
- Points in the alpha-hole primary gate with only 1 supporting frame are always dropped. It's tempting to keep them because the alpha gate is the "safe" path, but a single-frame alpha-hole candidate is exactly the failure mode this gate exists to catch.

#### 5.6.3 Diagnostics

Per-frame record gains:

```python
num_candidates_pre_consistency: int
num_candidates_post_consistency: int
consistency_drop_rate: float
avg_support_count: float
```

Pipeline-level:

```python
low_support_regions: list[BoundingBox]    # spatial clusters of high-drop-rate
trajectory_coverage_warning: bool         # True if > 30% of candidates dropped
```

`trajectory_coverage_warning` is a user-facing signal. When True, the pipeline's output message recommends re-running OmniRoam with `trajectory_mode="all"` or a gap-directed custom trajectory. **We do not auto-retry.** Retries complicate the pipeline state machine and hide problems the user should see.

#### 5.6.4 Implementation cost

KDTree build over ~500k points + radius queries: ~1–3 seconds on CPU. Small relative to Stage 1 depth estimation. Open3D's `KDTreeFlann` or SciPy's `cKDTree` both work. Unit test `test_cross_frame_consistency.py` synthesizes a scene where 1 frame sees a fake surface no other frame sees → verify it is dropped, then synthesizes 3 frames that all see the same real surface → verify all kept.

### 5.7 Stage 4: Aggregation

Survivors of the consistency gate are voxel-downsampled at the base splat's local gaussian spacing:

```python
def aggregate_candidates(
    survivors: SurvivingPoints,
    base_model: GaussianModel,
    voxel_size: float | None = None,
) -> AggregatedCandidates:
    """
    If voxel_size is None, estimate from base splat:
      voxel_size = median(kNN_distance(base_model.positions, k=5))
    """
```

Use Open3D's `VoxelDownSample` with per-point attribute averaging for color (SH_0) and majority vote for `hole_mode`. Preserve `source_frame_idx` as a list per voxel for provenance.

Expected output: ~50k–500k new gaussians depending on hole extent. If output exceeds `MAX_NEW_GAUSSIANS` (default 2M), increase voxel size by 1.5× and re-aggregate until under budget.

### 5.8 Stage 5: Gaussian Initialization

Inject aggregated points as new gaussians into the base model. The critical requirement is **statistical parity** with DA360-generated gaussians:

```python
def initialize_hole_gaussians(
    aggregated: AggregatedCandidates,
    base_model: GaussianModel,
    config: GeometricRefineConfig,
) -> GaussianModel:
    """
    Returns a new GaussianModel = base_model + new gaussians.
    Non-new parameters are copied byte-for-byte from base_model.
    """
```

New gaussian parameters:

| Parameter | Value | Rationale |
|---|---|---|
| `position` | voxel centroid | direct from aggregation |
| `scale` (3 axes) | `k · kNN_distance` (k=0.75, isotropic) | matches DA360 generator heuristic in `spag4d/generators/da360.py` |
| `rotation` | identity quaternion | DA360 convention; surface-normal alignment adds complexity without visual payoff |
| `opacity` (logit) | inverse_sigmoid(0.35) | moderate — color polish settles final value |
| `SH degree 0` | averaged RGB from contributing frames, sRGB→linear | direct color |
| `SH degrees 1–3` | zero | will be optimized in color polish if enabled |

**Provenance metadata (aggregation-aware).** The existing `GaussianProvenance` per-point metadata gains new fields:

```python
@dataclass
class GaussianProvenance:
    # existing
    source: Literal["base_panorama", "omniroam_geometric", ...]
    # revised for geometric refine (aggregation-aware)
    primary_source_frame_idx: int | None = None  # frame contributing closest to voxel centroid
    support_count: int = 0                        # distinct frames contributing to this voxel
    alignment_scale: float | None = None          # from primary source frame
    alignment_shift: float | None = None          # from primary source frame
    hole_mode: Literal["alpha", "disoccl"] | None = None  # majority vote across contributors
    positional_std: float | None = None           # voxel-internal position spread
```

PLY custom attributes:

```
property int   primary_source_frame_idx
property uchar support_count              # clamped to 255
property float alignment_scale
property float alignment_shift
property uchar hole_mode
property float positional_std
```

Full per-voxel contributor lists live in an optional sidecar `output.provenance.json` keyed by gaussian index, written only when `config.write_diagnostics=True`. This keeps the PLY small while preserving full traceability for debugging.

### 5.9 Stage 6: Color Polish (optional, default on)

Color polish is a short, tightly scoped finetune that addresses the only remaining weakness: new gaussians inherit their colors from OmniRoam, whose tone/white-balance/exposure don't exactly match the input panorama. Without polish, boundary seams are occasionally visible.

```python
@dataclass
class ColorPolishConfig:
    steps: int = 500
    lr: float = 1e-3
    sh_degree_max: int = 1     # 0, 1, or 2
    ssim_weight: float = 0.2
    early_stop_patience: int = 50
```

```python
def color_polish(
    model: GaussianModel,
    panorama_path: Path,
    base_depth: np.ndarray,
    new_gaussian_mask: np.ndarray,  # (N,) bool
    config: ColorPolishConfig,
) -> GaussianModel:
```

Constraints:

- **Only the new gaussians' SH coefficients receive gradient.** All other parameters (positions, scales, rotations, opacities, old SH) have `requires_grad = False`.
- **SH degree is configurable**, default 1. Degree 0 is fastest and avoids overfitting; degree 1 handles most boundary seams; degree 2 is available for cases where view-dependent effects from contributing OmniRoam frames create residual seams, but adds 15 coefficients per gaussian and risks overfitting to the single panorama viewpoint. The benchmark in §10.3 measures degree 1 vs degree 2 on the same scenes; default raises only if data supports it.
- **Only the panorama pose is used as supervision.** No OmniRoam frames. The loss is L1 + 0.2 · SSIM between rendered ERP from the panorama pose and the input panorama.
- **Short:** 500 iterations, AdamW at `lr=1e-3`, cosine decay to zero, early stop after 50 iterations of no improvement.

Because only new gaussians update, and only the ones visible from the panorama receive any gradient signal, this pass touches roughly the **boundary** gaussians — exactly the ones where color discontinuity would show. Interior hole gaussians (visible only from off-center OmniRoam poses) keep their averaged OmniRoam colors.

**Instrumentation:** color polish logs boundary seam L1 delta (computed over a thin ring of pixels around the new-gaussian region's projection in the panorama) before and after, so the impact of degree selection is measured rather than assumed.

Runtime: ~2 minutes on A6000 for a typical splat.

Disable with `config.color_polish_steps = 0` if you prefer pre-correction via histogram matching on OmniRoam frames before Stage 1 (alternate path documented in §9).

## 6. Configuration Surface

### 6.1 `GeometricRefineConfig`

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class GeometricRefineConfig:
    # Enable / backend
    enabled: bool = True
    depth_generator: Literal["da360", "dap", "sharp360"] = "da360"
    # ^ should usually match the generator used to build the base splat

    # OmniRoam (reuses existing OmniRoamConfig semantics)
    trajectory_mode: str = "auto"
    upscale_backend: Literal["none", "seedvr2"] = "seedvr2"
    max_frames: int = 81

    # Alignment
    align_mode: Literal["scale_shift", "scale_only"] = "scale_shift"
    alpha_confident_threshold: float = 0.90
    depth_gradient_threshold: float = 0.15
    min_inlier_fraction: float = 0.20

    # Hole filtering (per-frame)
    alpha_hole_threshold: float = 0.30
    depth_disoccl_margin_ratio: float = 0.02

    # Cross-frame consistency
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)

    # Aggregation
    voxel_size: float | None = None  # None -> auto from base splat kNN
    max_new_gaussians: int = 2_000_000

    # Color polish
    color_polish: ColorPolishConfig = field(default_factory=ColorPolishConfig)

    # Diagnostics
    write_diagnostics: bool = True
    save_per_frame_renders: bool = False  # debug only; large
```

### 6.2 Top-level entry point

```python
def refine_splat_geometric(
    ply_path: Path,
    panorama_path: Path,
    depth_map: np.ndarray,
    config: GeometricRefineConfig,
    output_path: Path | None = None,
) -> GeometricRefineResult:
    """
    If output_path is None, overwrites ply_path with a backup.

    Returns:
      GeometricRefineResult with:
        - output_ply_path
        - num_new_gaussians
        - per_frame_diagnostics: list[FrameDiagnostics]
        - pipeline_diagnostics: PipelineDiagnostics
        - total_runtime_seconds
    """
```

Matches the signature style of `refine_splat_v2` for drop-in compatibility. CLI:

```bash
python -m spag4d convert panorama.jpg output.ply \
    --refine --refine-backend geometric \
    --geometric-trajectory auto \
    --geometric-voxel-size auto
```

## 7. Data Structures and PLY Round-Tripping

The base splat's PLY format needs to round-trip provenance metadata. Current SPAG-4D writes standard 3DGS PLY (position, scale, rotation, opacity, SH). We extend the custom attribute block:

```
element vertex N
property float x
property float y
property float z
...existing 3DGS attrs...
property uchar source_kind                    # 0=base_panorama, 1=omniroam_geometric, ...
property int   primary_source_frame_idx       # -1 for base
property uchar support_count                  # 0 for base
property float alignment_scale                # NaN for base
property float alignment_shift                # NaN for base
property uchar hole_mode                      # 0=none, 1=alpha, 2=disoccl
property float positional_std                 # 0.0 for base
```

Backward compatible: existing 3DGS viewers ignore unknown properties. SPAG-4D's PLY reader gains an optional `load_provenance=True` kwarg that populates a parallel `GaussianProvenance` structure on `GaussianModel`. The optional `output.provenance.json` sidecar holds full per-voxel contributor lists for diagnostics.

## 8. Failure Modes and Diagnostics

Each frame writes a diagnostic record:

```python
@dataclass
class FrameDiagnostics:
    frame_idx: int
    alignment: AlignmentResult
    num_candidates: int
    num_kept_alpha: int
    num_kept_disoccl: int
    num_candidates_pre_consistency: int
    num_candidates_post_consistency: int
    consistency_drop_rate: float
    avg_support_count: float
    skipped: bool
    skip_reason: str | None
    wall_time_seconds: float
```

Pipeline-level:

```python
@dataclass
class PipelineDiagnostics:
    total_frames: int
    frames_skipped: int
    total_new_gaussians: int
    low_support_regions: list[BoundingBox]
    trajectory_coverage_warning: bool
    scale_consistency_std_ratio: float       # std(s_i) / mean(s_i) across frames
    color_polish_seam_delta: float | None
    total_runtime_seconds: float
```

Aggregated to `diagnostics.json` alongside the output PLY.

**Expected failure modes and responses:**

| Failure | Detection | Response |
|---|---|---|
| Frame alignment diverges | `inlier_fraction < 0.2` or `residual_p95 / median_depth > 0.25` | Skip frame, log |
| `s_i` highly inconsistent across frames | std > 10% of mean post-hoc check | Warn; suggest `align_mode_progressive=False` if it was on |
| Voxel downsample produces > `max_new_gaussians` | Output size check | Increase voxel size 1.5× and retry up to 3 times |
| Cross-frame gate drops > 30% of candidates | Pipeline-level check | Set `trajectory_coverage_warning=True`, surface to user |
| Color polish oscillates | Loss increases for 50 consecutive iterations | Early-stop, log |
| OmniRoam trajectory doesn't cover holes | Post-pipeline: measure remaining alpha<0.3 area | Warn; suggest different `trajectory_mode` |
| Depth network fails on a frame (NaN, all zeros) | Input validation in Stage 1 | Skip frame, log |

**Visual diagnostics** (optional, behind `save_per_frame_renders=True`): per-frame PNG of the base splat render, the OmniRoam frame, the validity mask, the kept-point projection, and the alignment residual heatmap. Useful for debugging single-frame failures; off by default because it writes ~800MB per run.

## 9. Alternate Paths and Toggleable Behaviors

**Histogram pre-matching instead of color polish.** For users who want fully deterministic output or don't want the brief training pass, set `config.color_polish.steps = 0` and add a pre-matching step that performs per-frame histogram matching of OmniRoam frames toward the input panorama's color distribution on a sky-masked subset before Stage 1.

**Global alignment instead of per-frame.** If post-hoc analysis shows `s_i` std < 5% of mean across the first 10 frames, switch to a single global `(s, t)` solved from those frames and apply to the rest. Saves ~20% total runtime; failure-prone on trajectories that actually need per-frame.

**SHARP 360 only on near-field frames.** SHARP 360's gaussians are most valuable on frames with strong near-field disocclusions; farther-from-origin frames don't benefit enough to justify SHARP's higher cost. Mixed-backend mode lets DA360 handle bulk frames and SHARP handle close-in ones.

## 10. Validation Plan

### 10.1 Unit tests (`tests/refine/geometric/`)

- `test_depth_convention.py` — verifies the `is_nearer_than_rendered` helper and that radial-form depth buffers are rejected loudly
- `test_depth_align_synthetic.py` — known scale/shift applied to ground-truth depth, verify recovery within 0.5%
- `test_depth_align_robust.py` — inject 20% outliers, verify Huber solve still converges
- `test_validity_mask.py` — sky/specular/gradient/erosion behave as expected on synthetic inputs
- `test_hole_filter_alpha_gate.py` — synthetic alpha buffer, verify only low-alpha regions pass primary filter
- `test_hole_filter_disoccl.py` — synthetic depth buffer with foreground occluder, verify parallax points pass secondary filter via the helper
- `test_cross_frame_consistency.py` — single-frame invention dropped; multi-frame agreement preserved; disoccl strict mode requires N+1
- `test_voxel_aggregate.py` — fixed input, verify deterministic output count
- `test_gaussian_init_stats.py` — sample DA360 base splat, verify new gaussians' scale distribution is within 1 std of base

### 10.2 Integration tests

- `test_hole_masked_ground_truth.py`: take a complete, dense 3DGS scene (e.g. from a scanned interior), synthetically mask a region to create "holes", run the full geometric refine, measure PSNR/SSIM **and** the geometry-stability metrics below on the filled region.
- `test_parity_with_base_generator.py`: run the geometric refine on a base splat with an **empty** OmniRoam input (zero frames). Expected output: identical to input (passthrough). Catches accidental mutation of base gaussians.

### 10.3 Qualitative + quantitative benchmark

Three representative panoramas from Cedar's test set:

1. Cave Junction creek cabin interior (strong near-field parallax)
2. Coastal exterior with mid-range occluders
3. High-ceiling interior with balcony occlusion

For each:

- Base splat (no refine)
- `refine_splat_v2` (current OmniRoam photometric)
- `refine_splat_geometric` with `color_polish.sh_degree_max=1`
- `refine_splat_geometric` with `color_polish.sh_degree_max=2`
- GSFix3D refine

Render 10 novel-view walkthroughs per splat, collect side-by-side videos, measure:

**Image quality:**
- PSNR/SSIM on held-out reference views (if available)
- Subjective hole-fill cleanliness (rank order)

**Geometry stability (new in v2):**
- **Duplicate geometry rate (DGR)**: fraction of new gaussians within `0.5 * local_kNN_distance` of an existing base gaussian with similar color (SH_0 L2 < 0.1). Target: < 2%.
- **Support-frame depth variance (SFDV)**: per new gaussian, std of contributing frames' unaligned depth-at-projection values normalized by the median, reported at the 95th percentile across the scene. Target: < 10%.
- **Held-out occlusion correctness (HOC)**: from a held-out viewpoint not in OmniRoam's trajectory, count new gaussians that appear in front of correctly-rendered base surfaces from the held-out view (occluding things they shouldn't). Target: < 1%.

**Cost:**
- Runtime and peak VRAM

Target: geometric refine beats v2 on hole-fill cleanliness and all three geometry-stability metrics, matches or beats GSFix3D, within 1.5× runtime of v2 (G5 target).

## 11. Implementation Phases

| Phase | Scope | Days |
|---|---|---|
| M1 | `depth_align.py` + validity mask + sky/specular masks + `depth_convention.py` + unit tests | 2 |
| M2 | `render_utils.py` — cube-face ERP composition for depth/alpha rendering | 1 |
| M3 | `hole_filter.py` — alpha gate + disocclusion gate via `is_nearer_than_rendered` | 1 |
| M3.5 | `consistency.py` — cross-frame mutual support filter + unit tests | 1 |
| M4 | `aggregate.py` + `init_gaussians.py` + PLY provenance round-trip + sidecar JSON | 2 |
| M5 | `pipeline.py` end-to-end + config + CLI | 1 |
| — | **First sanity-check milestone: run on 1 real scene, inspect** | — |
| M6 | `color_polish.py` with configurable SH degree and seam metric logging | 1 |
| M7 | `diagnostics.py` full report + trajectory coverage warnings | 0.5 |
| M8 | Integration tests (synthetic hole ground truth, empty-OmniRoam passthrough) | 1 |
| M9 | Qualitative + quantitative benchmark vs v2 and GSFix3D, including DGR/SFDV/HOC metrics | 2 |

**Total v1: ~12.5 days.** Critical path to first-real-scene sanity check: M1 → M5 = ~8 days.

The earliest useful output is after **M5**: a working geometric refine that can run on real data and visibly fill holes, even before color polish and full diagnostics land. This is the right point to sanity-check the approach on a single real scene before investing in the rest.

## 11bis Deferred to v1.1

These were specified in v1 but cut from the v1 critical path to keep the first ship narrow and debuggable. Ship v1, measure, then decide whether each is needed.

### ICP pose refinement

Point-to-plane ICP on the alignment overlap region to remove residual SE(3) misregistration after scale/shift solve. Specification:

```python
def icp_refine_pose(
    candidate_points: np.ndarray,        # (K, 3) from validity-mask region only
    base_points_nearby: np.ndarray,      # (L, 3) base splat points near pose
    max_iterations: int = 3,
    max_correspondence_distance: float = 0.05,
) -> tuple[np.ndarray, ICPStats]:
```

Two-iteration fixed schedule: align → ICP → re-align. Discard correction if translation > 10% of median base extent or rotation > 5°. Runtime to be measured; expect low-tens-of-ms per frame on Open3D.

**Decision criterion**: enable in v1.1 if M9 benchmark shows a measurable HOC or SFDV improvement when ICP is patched in.

### Base splat provenance filter

Tertiary hole signal: regions where the base splat's `GaussianProvenance` flags `background_leak` (a known DA360 failure mode on panoramas with near-field occluders) treated as holes even at higher alpha. Requires per-gaussian source fidelity from the base generators that's not yet reliable enough to drive geometry injection.

**Decision criterion**: enable in v1.1 once base generators emit provenance flags with measured precision/recall on a labeled set.

### Trajectory-aware scheduling

Skip Stage 1 depth estimation on frames that see no holes, per a pre-pass that renders only the base splat alpha from each pose and counts low-alpha pixels. Saves time on scenes where most holes are in one hemisphere.

**Decision criterion**: enable if Stage 1 is the runtime bottleneck in M9 measurements.

### SH degree 2 as color polish default

Currently configurable, default 1. Raise default to 2 if M9 benchmark shows degree 2 reduces boundary seam L1 by > 20% without introducing novel-view overfitting artifacts.

## 12. Open Questions

1. **Does the existing `spag4d.generators.da360` expose its unprojection math as a reusable helper, or is it embedded in a monolithic pipeline function?** If monolithic, refactor it out during M1. Non-trivial if it carries hidden state.
2. **How does SHARP 360 handle per-face seam alignment for ERP frames that aren't the original panorama?** SHARP's base-splat path uses DA360 to align face depths; we'd use the same mechanism per OmniRoam frame, but it roughly doubles Stage 1's runtime. Worth benchmarking before deciding whether SHARP 360 is a viable geometric refine backend or should stay DA360/DAP-only.
3. **Should `depth_map` (the one used to build the base splat) be stored in a sidecar JSON next to the PLY, or required as a separate input?** Currently `refine_splat_v2` requires it as a separate argument. **Recommendation: sidecar JSON**, loaded automatically if present and falling back to the explicit argument if not.
4. **OmniRoam v3 compatibility.** Wrap all OmniRoam calls in `spag4d/refine/geometric/omniroam_adapter.py` with a fixed internal contract of `(frames, poses)` and version-pin against OmniRoam's commit hash in `pyproject.toml`.
5. **Sky segmenter availability.** Does SPAG-4D already have a sky segmenter that can be reused for the validity mask, or do we need the HSV-heuristic fallback?

## 13. Summary

This design replaces photometric distillation of OmniRoam frames with a geometric fusion pipeline that extracts depth from each OmniRoam frame using the same backend that built the base splat, aligns those per-frame depths to the base splat's rendered geometry via robust scale/shift solve, and injects only the hole-region points as new gaussians **after** a cross-frame mutual-support gate confirms multiple frames independently agree on each point. A short SH-only color polish handles residual tone mismatch at boundaries. The approach is strictly additive on well-observed regions, inherits statistical parity with the base splat by construction, and needs no new trained models or dependencies beyond promoting `open3d` to a hard requirement for the geometric refine path.

The critical risk is depth alignment instability — if `(s_i, t_i)` varies wildly across frames, single-frame solves can fit overlap regions while extrapolating wrong geometry into hole regions. The cross-frame consistency gate in §5.6 is the primary defense: any candidate that lacks corroboration from at least 2 (or 3 for disocclusion candidates) other frames is dropped before aggregation. The pipeline also surfaces `trajectory_coverage_warning` when drop rates suggest the trajectory itself was insufficient, letting the user re-run with a better trajectory rather than silently accepting degraded output.

ICP pose refinement and base-splat provenance filtering are deferred to v1.1, narrowing the v1 critical path to alignment + per-frame filter + cross-frame consistency + aggregation + injection + color polish. The earliest useful output is after M5 (~8 days), at which point the approach can be sanity-checked on a single real scene before the remaining quality and diagnostic work lands.

Expected outcome: cleaner parallax disocclusion repair than `refine_splat_v2`, comparable or better than GSFix3D, with no photometric drift risk, measurable geometry-stability metrics (DGR, SFDV, HOC) instead of relying purely on subjective image-quality assessment, and a much smaller surface of tuning knobs than the current `tier2_weight` heuristic.
