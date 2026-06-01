# SPAG-4D × PaGeR Integration — Design Document

**Status:** Draft v1
**Author:** Cedar Connor
**Target:** `spag4d` core conversion pipeline
**Scope:** Add PaGeR as a fourth depth/geometry backend, slotting alongside DA360 and DAP, with optional sky-mask and surface-normal byproducts wired into existing filter and refine stages.

---

## 1. Summary

PaGeR (Panoramic Geometry Reconstruction, ETH/Google/Meta) is a single-forward-pass panoramic geometry model that outputs **scale-invariant depth, metric depth, surface normals, and a sky mask** from one equirectangular image. SPAG-4D already consumes exactly two of those four outputs through separate code paths: depth (DA360/DAP → `spag_converter.py`) and an auto-computed sky cutoff (`scene_analysis.py` percentile heuristic). PaGeR can replace the depth estimator with a stronger one *and* hand SPAG-4D a real sky mask and per-pixel normals it currently has to infer or do without.

This is a **drop-in generator addition**, not a pipeline rewrite. The integration mirrors the existing `da360_model.py` / `dap_model.py` contract: a model wrapper that returns a depth array which `spag_converter.py` projects into Gaussians. The new surfaces (sky mask, normals) are additive and optional — every existing code path keeps working untouched if they're ignored.

### Why bother

On the published benchmarks PaGeR beats DAP on real outdoor scenes by roughly 2× error reduction (ZüriPano scale-invariant AbsRel 9.36 vs DAP 19.86; δ₁ 94.75 vs 72.09). Outdoor is precisely the regime where your single-pano disocclusion problem is worst, so cleaner depth at the front of the pipeline means fewer holes for GSFix3D/OmniRoam to repair downstream. The normals output is a free upgrade to floater pruning and grazing-angle clipping. The sky mask replaces a percentile guess with a learned segmentation.

### Honest caveats (decide before building)

1. **Architecture ambiguity.** PaGeR's project page describes a multi-view-foundation build (Depth Anything 3, ViT-Giant, cubemap faces). The GitHub README describes a single-step diffusion model built on Marigold. These are different dependency surfaces. **Verify which checkpoints actually ship before committing** — see §9.
2. **Metric depth is the harder, weaker mode.** PaGeR's metric numbers are good but not a blowout (it loses ~1 of 9 metric metrics to DAP). Its *scale-invariant* depth is where it dominates. SPAG-4D's spherical projection cares about geometric consistency more than absolute scale (you already auto-fit depth range in `scene_analysis.py`), so the scale-invariant model is likely the right default — but you lose DAP's metric-radial property unless you use PaGeR's metric head.
3. **Training data (PanoInfinigen) is unreleased.** You can only run it zero-shot for now. No fine-tuning to your render statistics yet.
4. **Benchmarks skew indoor/urban.** Matterport, Stanford2D3DS, Zürich. Validate on your actual content (natural landscapes, hospitality interiors) before trusting it as default.
5. **Radial vs Z depth convention.** DAP outputs metric *radial* depth (distance along ray); your converter already handles this. PaGeR's depth convention must be confirmed and converted to match whatever `spag_converter.py` expects, or you'll get a subtly warped scene. This is the single most likely silent-failure bug.
6. **Resolution ceiling well below your render targets.** Both PaGeR variants cap output depth resolution far below SPAG-4D's 8K-input / 12K–16K-output ambitions, and the two architectures cap differently:
   - **DA3 multi-view variant:** input resolution is *decoupled by design* — the panorama is always projected to a fixed 6×504×504 cubemap, so it accepts any input but the depth signal bottlenecks at 504² per face, assembling to roughly a 2K ERP depth map. ~0.5 s / ~12.8 GB for a 2K pano.
   - **Diffusion variant:** the `PaGeR-depth` card states support for large resolutions up to ~3K, one-step regime.
   This is **less limiting than it sounds for SPAG-4D specifically**: depth defines *where* Gaussians sit, the source panorama (sampled independently at full res in `spag_converter.py`) defines their *color*. A 2K–3K depth map + 8K color source is a sound pairing, and DA360/DAP don't run at 8K either. The real cost is **softened geometry at depth discontinuities** — thin railings, foliage edges, foreground silhouettes — which a 2K depth map rounds off. For hospitality interiors/exteriors this is mostly fine; for foliage-heavy natural landscapes (Country Hills creek captures) it's where you'd see it, and exactly where post-projection hole rate matters most. **Mitigation: upsample the depth map to working resolution before projection — see §3.1.**

---

## 2. Where it fits in the current architecture

```
                          CURRENT
360 ERP panorama
  ├─ DA360 (da360_model.py) ──┐
  ├─ DAP   (dap_model.py)   ──┤→ depth (H,W) ─→ scene_analysis.py ─→ spag_converter.py ─→ scene_filter.py ─→ ply_writer.py
  └─ SHARP360 (sharp360.py) ──┘   (uses DA360 internally for alignment)

                          PROPOSED
360 ERP panorama
  ├─ DA360 ───────────────────┐
  ├─ DAP   ───────────────────┤
  ├─ SHARP360 ────────────────┤
  └─ PaGeR (pager_model.py) ──┘→ depth (H,W) ──→ scene_analysis.py ──→ spag_converter.py ──→ scene_filter.py ──→ ply_writer.py
       │                                              ▲                                          ▲
       ├─ sky_mask (H,W) bool ────────────────────────┘ (replaces percentile sky cutoff)        │
       └─ normals (H,W,3) ──────────────────────────────────────────────────────────────────────┘ (feeds grazing-angle + outlier pruning)
```

PaGeR enters as a peer of DA360/DAP — a depth source. The difference is it *also* emits two side-channels that two downstream stages can consume if present and ignore if absent.

---

## 3. Module-level design

### 3.1 New file: `spag4d/pager_model.py`

Mirror the structure of `da360_model.py`. Public surface:

```python
# spag4d/pager_model.py
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class PaGeRResult:
    depth: np.ndarray            # (H, W) float32 — Z or radial, see .depth_convention
    depth_convention: str        # "radial" | "z"  — set explicitly, never guess downstream
    metric: bool                 # True if metric head used, False if scale-invariant
    sky_mask: np.ndarray | None  # (H, W) bool — True = sky, None if head unavailable
    normals: np.ndarray | None   # (H, W, 3) float32 world or camera frame, unit length
    normals_frame: str           # "camera" | "world"
    native_resolution: tuple     # (H, W) the model actually produced before any upsample
                                 #   DA3 variant: ~(1024, 2048); diffusion: up to ~(1536, 3072)
                                 #   Recorded so the PLY header / logs note real depth fidelity.

class PaGeRModel:
    """Panoramic geometry estimator. Same role as DA360Model / DAPModel."""

    def __init__(self, device="cuda", variant="depth",
                 enable_xformers=False, checkpoint=None,
                 target_resolution=None):
        # variant: "depth" (scale-invariant), "metric-depth", "normals",
        #          plus indoor specializations. See §4 checkpoint matrix.
        # enable_xformers default FALSE — see §9 Windows note.
        # target_resolution: (H, W) to upsample depth/sky/normals to before
        #   returning. None = leave at native. core.py passes the working
        #   resolution that spag_converter expects (typically the source
        #   panorama resolution / stride). See §3.1 resolution contract.
        ...

    def load(self):
        """Lazy-load weights. Match da360_model.py lazy pattern so the
        registry can construct without paying VRAM cost until convert."""
        ...

    @torch.inference_mode()
    def infer(self, panorama: np.ndarray) -> PaGeRResult:
        """panorama: (H, W, 3) uint8 or float. Returns PaGeRResult.
        Internally: ERP → (cubemap if DA3 variant) → forward pass →
        reproject heads back to ERP → record native_resolution →
        convert depth convention → upsample to target_resolution.

        Upsample rules (critical — see resolution contract below):
          - depth:   bilinear. Upsampling depth is safe; it's a smooth field.
          - sky_mask: nearest. Never bilerp a boolean mask.
          - normals: bilinear THEN re-normalize to unit length per pixel
                     (interpolation breaks unit-length); re-normalize after.
        """
        ...
```

**Resolution contract (the part that keeps stride math honest):**

PaGeR produces depth at its native ceiling — ~2K ERP for the DA3 cubemap variant (fixed 504² faces), up to ~3K for the diffusion variant — which is *below* SPAG-4D's working resolution on high-res captures. The rest of the pipeline (`scene_analysis` percentiles, `spag_converter` stride sampling, `scene_filter` neighborhoods) assumes depth and source panorama share a coordinate grid. To preserve that invariant **without disturbing the DA360/DAP code path**, `PaGeRModel` upsamples depth/sky/normals to `target_resolution` *inside* `infer()`, before returning. Then `spag_converter` strides over a full-resolution depth map exactly as it does for DA360, and color sampling from the full-res source panorama lines up pixel-for-pixel.

Two honest notes on this:
- Upsampling does **not** recover lost geometric detail — a 2K depth map bilinear-upsampled to 8K is still 2K worth of structure. It only restores grid alignment so stride/color/filter math is consistent. The detail-at-discontinuities limitation (caveat 6) stands.
- `native_resolution` is recorded on the result so logs and the PLY header reflect the *real* depth fidelity, not the post-upsample shape. Don't let the upsample hide what actually happened.



**Key decisions baked into the dataclass:**

- `depth_convention` is explicit and travels with the data. Do **not** let any downstream module assume. The most likely PaGeR-vs-SPAG mismatch bug is radial/Z confusion; making the convention a required field forces a conversion decision at the boundary.
- `sky_mask` and `normals` are nullable. If you run the `depth` checkpoint only, they come back `None` and nothing downstream breaks.
- `normals_frame` matters because `spag_converter.py` builds world-frame ray directions; normals must be rotated into the same frame before any geometric filter uses them.

### 3.2 Generator registration in `core.py`

`core.py` is the pipeline orchestrator that dispatches on `generator`. Add `"pager"` to the dispatch table. Because PaGeR is depth-based like DA360/DAP, it reuses the **entire** depth→Gaussian path (`scene_analysis` → `spag_converter` → `scene_filter` → `ply_writer`). The only new wiring is passing the optional sky mask and normals through.

```python
# core.py — inside the generator dispatch
if generator == "pager":
    H, W = panorama.shape[:2]                      # working/source resolution
    model = PaGeRModel(device=self.device,
                       variant=opts.pager_variant,
                       enable_xformers=opts.enable_xformers,
                       target_resolution=(H, W))   # upsample inside infer() — §3.1
    model.load()
    result = model.infer(panorama)

    # surface real depth fidelity, not the post-upsample shape
    log.info("PaGeR native depth %s → upsampled to %s",
             result.native_resolution, result.depth.shape)

    depth = _to_converter_convention(result.depth, result.depth_convention)

    # sky mask: prefer learned mask over percentile heuristic
    sky_mask = result.sky_mask  # may be None

    # normals: rotate to world frame if needed, else None
    normals = _normals_to_world(result.normals, result.normals_frame) \
              if result.normals is not None else None

    scene = analyze_scene(depth, sky_mask=sky_mask)        # see §3.4
    gaussians = spherical_project(depth, panorama, scene)  # depth grid now matches source
    gaussians = filter_scene(gaussians, normals=normals)   # see §3.5
    write_ply(gaussians, out_path)
```

Passing `target_resolution=(H, W)` makes the depth grid match the source panorama before projection, so `spherical_project` strides over it identically to the DA360/DAP path — no special-casing in the converter.

### 3.3 CLI + API surface

Extend `cli.py` and the Python `SPAG4D.convert()` signature. Keep it parallel to existing flags:

```bash
# scale-invariant depth (recommended default for PaGeR)
python -m spag4d convert pano.jpg out.ply --generator pager

# metric head (drop-in replacement for DAP's metric property)
python -m spag4d convert pano.jpg out.ply --generator pager --pager-variant metric-depth

# indoor-specialized checkpoint
python -m spag4d convert pano.jpg out.ply --generator pager --pager-variant depth-indoor

# use the learned sky mask + normals for filtering
python -m spag4d convert pano.jpg out.ply --generator pager --pager-use-sky --pager-use-normals
```

```python
result = converter.convert("pano.jpg", "out.ply",
    generator="pager",
    pager_variant="depth",        # depth | metric-depth | normals | *-indoor
    pager_use_sky=True,
    pager_use_normals=True,
)
```

### 3.4 `scene_analysis.py` change — sky mask injection

Today `analyze_scene()` computes `sky_threshold` as a depth percentile (default 95th). Add an optional `sky_mask` parameter. When a real mask is supplied:

- Skip the percentile sky-cutoff entirely; use the mask to drop sky Gaussians directly.
- Still compute `depth_min`/`depth_max`/`orbit_radius` from the **non-sky** depth pixels only, which makes the auto depth-range fit far more robust outdoors (the percentile method gets dragged by sky pixels reading as max depth). This alone may be worth the integration for outdoor hospitality-property captures.

```python
def analyze_scene(depth, sky_mask=None):
    valid = ~sky_mask if sky_mask is not None else _percentile_sky(depth)
    d = depth[valid]
    depth_min = np.percentile(d, 1)
    depth_max = np.percentile(d, 99)
    ...
```

Backward compatible: `sky_mask=None` reproduces today's behavior exactly.

### 3.5 `scene_filter.py` change — normal-aware pruning

Two existing filters can use real normals instead of geometric proxies:

- **Grazing-angle clip** (currently `grazing_angle=65`): today this is inferred from the angle between the view ray and a locally estimated surface from depth gradients. With PaGeR normals you have the surface orientation directly — compute `dot(view_ray, normal)` per Gaussian and clip below the cosine threshold. Cleaner, no depth-gradient noise.
- **Outlier / floater pruning**: normals give a second signal — floaters tend to have normals inconsistent with their neighbors. Optional secondary pass; keep the existing spatial pruning as the primary.

Make this strictly additive: if `normals is None`, fall back to the current depth-gradient grazing-angle path. No regression risk.

### 3.6 PLY provenance (optional, low priority)

Your `refine/provenance.py` already tags Gaussian origin (`original` / `densified` / `omniroam` / `gap_seed`). No new provenance kind is needed — PaGeR-sourced base Gaussians are still `original`. But consider stashing the generator name and `pager_variant` in the PLY header comment block for reproducibility, since you now have four possible base generators.

---

## 4. Checkpoint matrix

PaGeR ships multiple heads. Map them to SPAG-4D variants:

| `--pager-variant` | HF checkpoint | Output used | Native res ceiling | Notes |
|---|---|---|---|---|
| `depth` (default) | `prs-eth/PaGeR-depth` | scale-invariant depth | diffusion ~3K / DA3 ~2K | Best generalization; pairs with auto depth-range fit |
| `metric-depth` | `prs-eth/PaGeR-metric-depth` | metric depth | ~2–3K | DAP replacement when you need absolute scale |
| `depth-indoor` | `prs-eth/PaGeR-depth-indoor` | scale-invariant depth | ~2–3K | Hospitality interiors |
| `metric-depth-indoor` | `prs-eth/PaGeR-metric-depth-indoor` | metric depth | ~2–3K | Indoor + metric |
| `normals` | `prs-eth/PaGeR-normals` | surface normals | ~2–3K | Standalone normals if you want them without re-running depth |
| `normals-s3d` | `prs-eth/PaGeR-normals-Structured3D` | surface normals | ~2–3K | Finetuned variant |

The native-res ceiling is the depth fidelity *before* the §3.1 upsample, and it depends on which architecture ships: the DA3 multi-view variant is fixed at 504² cube faces → ~2K ERP regardless of input; the diffusion variant's `PaGeR-depth` card claims support up to ~3K. Confirm the exact ceiling per checkpoint in Phase 0 and tighten this column — the ~2–3K entries are inferred, not yet measured.

If the unified multi-task model (one checkpoint → all four heads) is what actually ships, collapse this table to a single `--pager-variant unified` and select outputs by flag instead. **This is the §9 verification item.**

Follow the existing weight-download convention: extend `download-models` (`cli.py`) with `--model pager` and cache under a `pretrained/pager/` directory paralleling `pretrained/gsfix3d/`.

---

## 5. Dependency & environment impact

PaGeR's stack (Marigold lineage) is `torch` + `diffusers` + `transformers` + `omegaconf` + `einops` — you already have all of these from GSFix3D refinement (`pip install diffusers transformers ...`). If it's the DA3 multi-view variant instead, that's also pure PyTorch. **Either way, no new CUDA-compiled dependency** — unlike OmniRoam (WSL2) this stays native Windows.

- `--enable_xformers` is **optional** in PaGeR and is the one likely Windows pain point. Default it `False` in `PaGeRModel`; your A6000's 48 GB eats the memory difference without it via PyTorch SDPA.
- Add a `requirements-pager.txt` (parallel to `requirements-refine.txt`) pinning only what's not already covered, so core install stays lean.
- Vendor or submodule PaGeR's inference code under `spag4d/pager_arch/` mirroring `spag4d/da360_arch/` and `spag4d/sharp_arch/ml-sharp/`. Given the architecture ambiguity, **vendoring a pinned commit is safer than a live submodule** until the repo stabilizes (it's currently ~11 commits old).

### License note
PaGeR code is Apache-2.0 (clean, commercial-OK). **Model weights are RAIL++-M** — a responsible-AI license with use restrictions, *not* unrestricted commercial like DA360. This matters for Connor Hospitality / Sphere commercial work. It's a different status from your MIT core and the noncommercial SHARP/OmniRoam modules — update the README license table to a fourth row. Treat PaGeR as "check restrictions before commercial deliverable," closer to the SHARP/OmniRoam caveat than to the MIT core.

---

## 6. Implementation plan (phased)

**Phase 0 — Verify (½ day).** Resolve the architecture ambiguity (§9). Pull the actual shipped checkpoints, run PaGeR's own `inference.py` on 3–4 of your test panoramas (one outdoor hospitality exterior, one cabin interior, one natural landscape, the demo pano). Confirm depth convention (radial vs Z) by reprojecting a known-distance feature. **Do not write integration code until this is done** — the whole design pivots on it.

**Phase 1 — Depth-only backend (1 day).** Build `pager_model.py` returning depth only (`sky_mask=None`, `normals=None`). Register `"pager"` in `core.py`. Wire CLI/API flags. This gets you a working fourth generator that reuses the entire existing projection/filter/export path. Ship and A/B against DAP on your test set.

**Phase 2 — Sky mask (½ day).** Plumb `sky_mask` into `scene_analysis.py` (§3.4). Biggest practical win for outdoor auto depth-range. Gate behind `--pager-use-sky`.

**Phase 3 — Normals (1 day).** Plumb `normals` into `scene_filter.py` grazing-angle clip (§3.5). Validate the normals frame conversion carefully against `spag_converter.py`'s ray-direction frame. Gate behind `--pager-use-normals`.

**Phase 4 — Polish (½ day).** `download-models --model pager`, `requirements-pager.txt`, README fourth-generator row + license row, INSTALL.md note, a smoke test in `tests/`.

Total: ~3.5 focused days, front-loaded on verification.

---

## 7. Validation & acceptance

Reuse PaGeR's own eval harness *and* your downstream metric — the point isn't depth accuracy in isolation, it's fewer holes after projection.

- **Front-end depth:** run PaGeR vs DAP vs DA360 on your test panoramas; eyeball depth maps for ERP pole behavior and outdoor sky/horizon handling.
- **Downstream hole rate:** after `spag_converter` + `scene_filter`, render from the 36 evaluation cameras (`refine/camera_rig.py` already does this) and measure hole percentage *before refinement*. Acceptance: PaGeR base produces ≤ DAP base hole rate on outdoor scenes. This is the metric that actually matters for your pipeline.
- **Normals sanity:** colorize normals as RGB, confirm walls/ground read as coherent flat regions, not noise.
- **Regression guard:** a DA360 and a DAP conversion must produce byte-identical output before/after the integration (the additive-only design guarantees this; test it anyway).

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Depth convention mismatch (radial/Z) → warped scene | High | Explicit `depth_convention` field; Phase 0 reprojection check |
| Architecture ambiguity (DA3 vs diffusion) changes deps | Medium | Phase 0 verification gate before any code |
| Normals frame mismatch → wrong grazing clip | Medium | Validate against converter ray frame in Phase 3; additive fallback |
| RAIL++-M license blocks commercial use | Medium | Flag in README; treat like SHARP/OmniRoam noncommercial modules |
| xformers Windows install failure | Low | Default off; rely on SDPA |
| Zero-shot quality poor on natural landscapes | Medium | Phase 0 content test; keep DAP as default if PaGeR underperforms your scenes |
| Native depth res (2–3K) softens fine geometry vs 8K capture | Medium-High | Upsample-before-projection (§3.1) keeps grid aligned; accept detail loss at discontinuities or keep DAP for foliage-heavy scenes; log `native_resolution` |
| PanoInfinigen unavailable → no fine-tune | Certain (for now) | Accept zero-shot; revisit when dataset drops |

---

## 9. Open question to resolve first (BLOCKER)

**Which architecture actually ships?** The project page (cubemap + Depth Anything 3 multi-view foundation) and the GitHub README (single-step Marigold diffusion) describe different models. This determines:
- ERP→cubemap preprocessing (needed for DA3 variant, not for a diffusion ERP model)
- Whether one unified checkpoint emits all four heads or you load separate depth/metric/normals checkpoints
- The actual native output resolution per checkpoint (measure it — the §4 ceiling column is inferred) and therefore how much the §3.1 upsample is doing
- Exact dependency pins

Resolve by: cloning `prs-eth/PaGeR`, reading `inference.py` + `app.py` + the `configs/` YAMLs, and inspecting one downloaded checkpoint's config. Everything in §3–§4 adapts cleanly to either answer, but the preprocessing in `pager_model.infer()` and the checkpoint matrix in §4 must match reality before Phase 1.

---

## 10. Recommendation

Build Phase 1 (depth-only) first and A/B it against DAP on outdoor captures. If PaGeR's scale-invariant depth produces a measurably lower pre-refinement hole rate on your content — which the ZüriPano numbers predict — promote it to the outdoor default and proceed to Phases 2–3 for the sky/normals wins. Keep DA360 as the fast preview default and DAP as the metric-when-needed option. PaGeR becomes your *quality* depth backend, especially outdoors, without disturbing anything that already works.
