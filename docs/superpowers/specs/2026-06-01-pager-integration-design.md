# SPAG-4D × PaGeR Integration — Spec

**Status:** Approved design, ready for planning
**Date:** 2026-06-01
**Author:** Cedar Connor (design), refined against verified PaGeR sources
**Supersedes:** `docs/pager_integration_design.md` (Draft v1) — that draft's §9 blocker is now resolved and several of its assumptions are corrected here.
**Scope:** Add PaGeR as a fourth depth/geometry backend in `spag4d`, peer to DA360/DAP/SHARP360, with optional learned sky-mask and surface-normal byproducts wired into the existing scene-analysis and filter stages. Strictly additive — DA360/DAP/SHARP360 paths are untouched.

---

## 1. Summary

PaGeR ("Unified Panoramic Geometry Estimation via Multi-View Foundation Models", ETH Zürich / prs-eth) is a single-forward-pass panoramic geometry model that outputs scale-invariant depth, metric depth, world-frame surface normals, and a learned sky mask from one equirectangular (ERP) image. SPAG-4D currently consumes only depth (DA360/DAP) plus a percentile sky-cutoff heuristic. PaGeR replaces the depth estimator with a stronger one *and* hands SPAG-4D a real sky mask and per-pixel normals it currently infers or does without.

This is a drop-in generator addition. PaGeR enters as a peer depth source; its two side-channels (sky, normals) are additive and optional — every existing path keeps working untouched if they are absent.

### Why
On published benchmarks PaGeR roughly halves DAP's error on real outdoor scenes (ZüriPano scale-invariant AbsRel 9.36 vs DAP 19.86; δ₁ 94.75 vs 72.09). Outdoor is where SPAG-4D's single-pano disocclusion problem is worst, so cleaner front-of-pipeline depth means fewer holes for downstream geometric-refine to repair. Normals upgrade grazing-angle/floater pruning from depth-gradient proxies to a direct signal. The sky mask replaces a percentile guess with a learned segmentation.

---

## 2. Verified facts (Phase 0 — RESOLVED)

These resolve the Draft v1 §9 blocker. Sources: github.com/prs-eth/PaGeR (README, `inference.py`, `src/pager.py`, `src/utils/geometry_utils.py`, `requirements.txt`, `configs/`), arXiv 2605.26368, project page pager360.github.io, and HF repos `prs-eth/PaGeR`, `prs-eth/PaGeR-metric-depth`, `prs-eth/PaGeR-normals`.

| Question | Verified answer |
|---|---|
| **Architecture that ships** | **DA3 cubemap multi-view only.** `inference.py` builds `Pager` → `DepthAnything3` (ViT-Giant). No diffusers/UNet/VAE in the tree. Marigold is the *training* script only (not shipped). The "single-step diffusion" framing is obsolete. |
| **Checkpoints (exist + downloadable)** | Exactly three: `prs-eth/PaGeR` (**unified**, all heads, ~5.66 GB), `prs-eth/PaGeR-metric-depth`, `prs-eth/PaGeR-normals`. The draft's `PaGeR-depth`, `*-indoor`, `*-Structured3D` repos **do not exist** (404). |
| **Indoor vs outdoor** | Not a separate checkpoint — two scale heads *inside* the unified model, auto-selected at runtime by a CLIP ViT-B/32 router on the cubemap faces. No user flag needed. |
| **Output heads** | Unified checkpoint emits all four: SI depth, metric depth (SI × per-pano scale), **world-frame** normals (unit vectors), and a **learned sky** head (sigmoid logits). |
| **Depth convention** | Raw head = per-cube-face **log-Z**. Shipped `process_depth_output()` un-logs → applies scale → `z_depth_to_euclidean()` → sky-fill → `cubemap_to_erp()`, yielding **radial/Euclidean distance in ERP layout** — i.e. the DAP-style radial convention `spag_converter` already expects. Sky regions filled to MAX_DEPTH. |
| **Resolution** | Internal compute fixed at **6×504×504 cubemap** regardless of input. Output ERP size is a stitch argument (default 1024×2048, supports up to ~3K). Real geometric detail is capped at 504 px/face; upsampling restores grid alignment, not detail. |
| **Dependencies** | `torch>=2.0`, `torchvision`, `numpy<2` (hard pin — **already satisfied: venv has 1.26.4**), `scipy`, `einops`, `addict`, `safetensors`, `huggingface_hub`, `omegaconf`, `open_clip_torch`, `opencv-python`, `pytorch360convert`, `trimesh`, `tqdm`. **No diffusers, no transformers, no CUDA-compiled custom ops.** xformers optional (default off → SDPA). |
| **Platform** | Officially tested Linux / CUDA≥12.1. **Native Windows untested** (repo ~2 weeks old, 0 issues). Venv already has torch 2.5.1+cu121 / CUDA 12.1. Pure-PyTorch stack → native Windows expected to work; **must be confirmed by a Phase 1 smoke test.** |
| **License** | Code **Apache-2.0**. Weights **CC BY-NC 4.0 (non-commercial)** — inherited from the DA3 ViT-Giant backbone, **cannot be relicensed.** |
| **Inference API** | `Pager` class (`src/pager.py`). `get_intrinsics_extrinsics()` must be called once before first forward. `Pager.forward(rgb_cubemap, dtype, skip_heads)` takes an **ImageNet-normalized cubemap tensor `(B,6,3,504,504)`**, face order `[F,R,B,L,U,D]`, returns a raw dict (`depth`=log-Z/face, `normals`, `sky`=logits, `scale`=log-scale). Helpers: `erp_to_cubemap`/`cubemap_to_erp` (`src/utils/geometry_utils.py`), `process_depth_output`, `process_normals_output`. On the unified checkpoint, drop one scale head per call via `skip_heads={"scale_indoor"}` or `{"scale_outdoor"}` (CLI uses the CLIP router to pick). |

### Residual unknowns (carry into Phase 1, not blockers)
- (a) Native-Windows execution of `open_clip_torch` / `pytorch360convert` / DINOv2 attention.
- (b) Exact `erp_to_cubemap` signature (read the inverse `cubemap_to_erp` in full; verify before wiring).
- (c) World-frame normals axis/sign convention vs. `spag_converter`'s world ray frame (Phase 3 check).

---

## 3. Decisions (approved)

1. **License posture:** Build for evaluation. DA360/DAP remain the commercial-safe defaults; PaGeR is gated behind an explicit non-commercial warning in CLI output + README. Commercial use resolved later (negotiate, or use a commercial-safe backend for deliverables).
2. **Hosting:** Vendored in-process in the existing `.venv`, mirroring `da360_arch/` / `sharp_arch/`. xformers off. WSL2 is a documented fallback *only if* the Phase 1 smoke test fails on native Windows.
3. **API binding:** Map the sky mask onto the existing depth-backend return slot `predict() → (depth, mask)`; stash normals + native-resolution + convention on the model instance. No new `PaGeRResult` dataclass crossing into the pipeline (rejected as needing a core.py adapter for no gain).
4. **Weights cache:** `~/.cache/spag4d/` via `huggingface_hub` (DA360 convention), not `pretrained/pager/`.
5. **Metric depth:** Exposed as a `--pager-metric` toggle on the unified checkpoint (SI is the default), not separate variant strings.

---

## 4. Where it fits

```
360 ERP panorama
  ├─ DA360 (da360_model.py) ──┐
  ├─ DAP   (dap_model.py)     ┤→ depth (H,W) → compute_scene_defaults → spag_converter → scene_filter → ply_writer
  ├─ SHARP360 (sharp360.py) ──┤   (special-cased branch)
  └─ PaGeR (pager_model.py) ──┘
       │                            ▲                                  ▲
       ├─ sky_mask (H,W) bool ──────┘ (excludes sky from depth-range   │
       │                               percentile fit; drops sky)      │
       └─ normals (H,W,3) world ────────────────────────────────────────┘ (direct grazing-angle clip)
```

PaGeR is a depth source that *also* emits two channels two downstream stages consume if present, ignore if absent.

---

## 5. Module design (bound to the real API)

### 5.1 `spag4d/pager_arch/PaGeR/` — vendored upstream
Vendor a **pinned commit** of `prs-eth/PaGeR` (Apache-2.0 code only — no weights), mirroring `da360_arch/DA360/`. Vendoring (not a live submodule) because the repo is ~2 weeks old and unstable. Add `spag4d/pager_arch/__init__.py` exposing a `build_pager(...)` helper, paralleling `da360_arch.build_da360_model`. Strip the upstream `[app]`/gradio extras.

### 5.2 `spag4d/pager_model.py` — `PaGeRModel`
Matches the existing depth-backend contract (`DA360Model` / `DAPModel`), **not** the draft's `infer()→dataclass`:

```python
class PaGeRModel:
    @classmethod
    def load(cls, device=torch.device("cuda"), metric: bool = False,
             erp_out: tuple[int,int] | None = None) -> "PaGeRModel":
        """Lazy-load unified prs-eth/PaGeR weights from HF cache.
        metric: route the metric scale head (default SI depth).
        erp_out: ERP stitch size; None → derive from input at predict()."""

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """image: (H,W,3) uint8/float on device. Returns (depth, sky_mask).
        depth: (H,W) float32 radial/Euclidean meters (or SI units if not metric),
               upsampled to (H,W) bilinear.
        sky_mask: (H,W) bool, sigmoid(sky_logits) > 0.5, upsampled nearest.
        Side-effects stashed for core.py to pull:
            self.last_normals: (H,W,3) float32 world-frame unit, bilinear+renormalized
            self.native_resolution: (h,w) before upsample (~1024×2048)
            self.depth_convention: "radial"
            self.metric: bool
        """
```

Internals of `predict()`:
1. ERP → cubemap `(1,6,3,504,504)`, ImageNet-normalized, face order `[F,R,B,L,U,D]` (`erp_to_cubemap`).
2. Ensure `get_intrinsics_extrinsics()` was called once at load.
3. `Pager.forward(cubemap, dtype=fp16, skip_heads=…)` — skip the unused scale head (CLIP router picks indoor/outdoor).
4. `process_depth_output(...)` → radial ERP depth at `native_resolution`.
5. `process_normals_output(...)` → world-frame ERP unit normals.
6. `sigmoid(sky_logits) > 0.5` → bool ERP sky mask.
7. **Upsample to `(H,W)`**: depth bilinear, sky nearest, normals bilinear **then re-normalize to unit length** per pixel. Record `native_resolution`.

**Resolution contract:** Upsampling restores grid alignment so `spag_converter` strides over depth exactly as for DA360, and full-res color sampling lines up pixel-for-pixel. It does **not** recover geometric detail (capped at 504 px/face). `native_resolution` is recorded so logs/PLY header note real depth fidelity, not the post-upsample shape.

### 5.3 `core.py` — generator dispatch
Add an `active_generator == "pager"` branch alongside the existing `sharp360` special-case and the `da360`/`dap` depth path. It:
1. Constructs `PaGeRModel.load(device, metric=opts.pager_metric, erp_out=(H,W))` and caches it in `self._depth_models["pager"]`.
2. Calls `depth, sky_mask = model.predict(image_tensor)`; logs `native_resolution → (H,W)`.
3. Pulls `normals = model.last_normals` (or `None`).
4. Applies `--pager-use-sky` / `--pager-use-normals` gates (channel set to `None` if its flag is off).
5. Feeds the **existing** flow: `compute_scene_defaults(depth_np, image_height=H, sky_mask=sky_mask)` → `_run_spag_pipeline(...)` → `prune_outliers` / `prune_grazing_angle(..., normals=normals)` / `prune_sparse_regions` → `save_ply_gsplat(..., colors_linear=False)`.

No change to the SPAG projection or PLY paths. Optionally stash `generator="pager"` + metric/SI in the PLY header comment for provenance (low priority).

### 5.4 `scene_analysis.compute_scene_defaults` — sky injection
Add `sky_mask: np.ndarray | None = None`. When supplied:
```python
valid = (depth_map > 0.01) & np.isfinite(depth_map)
if sky_mask is not None:
    valid &= ~sky_mask
```
All percentiles (`depth_min`, `depth_max`, `sky_threshold`, `orbit_radius`) then fit over **non-sky** pixels — the main robustness win outdoors, where sky pixels otherwise drag the depth range. `sky_mask=None` reproduces today's behavior exactly. Sky Gaussians are dropped directly via the mask (sky-fill MAX_DEPTH pixels also exceed the fitted range).

### 5.5 `scene_filter.prune_grazing_angle` — normal-aware clip
Add `normals: np.ndarray | None = None`. When present, compute `cos = dot(view_ray, normal)` per Gaussian and clip below the cosine threshold directly, replacing the depth-gradient surface estimate. When `None`, fall back to the current depth-gradient path — no regression risk. Requires a Phase 3 axis/sign check: PaGeR normals are world-frame; confirm they share `spag_converter`'s world ray-direction frame (handedness, up axis) before trusting the dot product. Floater pruning via normal-neighbor consistency is an optional later add, not in scope here.

### 5.6 CLI / API
```
python -m spag4d convert pano.jpg out.ply --generator pager                 # SI depth, no side-channels
python -m spag4d convert pano.jpg out.ply --generator pager --pager-metric  # metric scale head
python -m spag4d convert pano.jpg out.ply --generator pager --pager-use-sky --pager-use-normals
```
Python: `converter.convert(..., generator="pager", pager_metric=False, pager_use_sky=True, pager_use_normals=True)`.
CLI prints a one-line **non-commercial license** warning whenever `--generator pager` is used.

### 5.7 Weights, deps, packaging
- `download-models --model pager` → `huggingface_hub.snapshot_download("prs-eth/PaGeR")` into `~/.cache/spag4d/pager/`. `PaGeRModel.load` resolves from that cache, downloading on first use if absent.
- `requirements-pager.txt` pins only what's not already present: `open_clip_torch`, `pytorch360convert`, `omegaconf`, `einops`, `addict`, `trimesh`. (`numpy<2` already satisfied.)
- README: fourth-generator row + fourth license row (CC BY-NC 4.0, "non-commercial — evaluation only"). INSTALL.md note on the optional requirements file and the WSL2 fallback.

---

## 6. Phased implementation

**Phase 0 — Verify.** ✅ Done (§2). Architecture, checkpoints, convention, deps, license resolved.

**Phase 1 — Smoke + depth-only backend.**
- Vendor `prs-eth/PaGeR` (pinned commit) under `spag4d/pager_arch/PaGeR/`; add `build_pager` wrapper.
- **Smoke gate:** run vendored PaGeR on 2–3 panoramas (one outdoor exterior, one interior, one natural landscape) on **native Windows** to confirm it runs, resolve the exact `erp_to_cubemap` signature, and reproject a known-distance feature to confirm radial output. If native Windows fails, switch to the WSL2 fallback before proceeding.
- Build `PaGeRModel` returning depth only (sky/normals stashed but unused).
- Register `pager` in `core.py`; wire `--generator pager` + `--pager-metric`.
- A/B vs DAP on outdoor captures (front-end depth eyeball + downstream hole rate).

**Phase 2 — Sky mask.** Plumb `sky_mask` into `compute_scene_defaults` (§5.4). Gate `--pager-use-sky`. Validate outdoor auto depth-range improvement.

**Phase 3 — Normals.** Plumb `normals` into `prune_grazing_angle` (§5.5) after the world-frame axis check. Gate `--pager-use-normals`. Colorize normals as RGB to confirm coherent walls/ground.

**Phase 4 — Polish.** `download-models --model pager`, `requirements-pager.txt`, README rows, INSTALL.md note, `tests/` smoke test, PLY-header provenance.

---

## 7. Validation & acceptance

- **Regression guard (must pass):** a DA360 and a DAP conversion produce byte-identical output before vs. after the integration. Guaranteed by the additive design; tested anyway.
- **Front-end depth:** PaGeR vs DAP vs DA360 on the test panoramas; eyeball ERP pole behavior and outdoor sky/horizon handling.
- **Downstream hole rate (the metric that matters):** after `spag_converter` + `scene_filter`, render from the 36 evaluation cameras (`refine/camera_rig.py`) and measure hole % *before refinement*. **Acceptance: PaGeR base ≤ DAP base hole rate on outdoor scenes.**
- **Sky mask:** outdoor depth range fit no longer dragged by sky; sky Gaussians removed.
- **Normals sanity:** RGB-colorized normals read as coherent flat regions, not noise; grazing clip removes the same or fewer correct surfaces vs. the depth-gradient path.

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Native-Windows execution fails (open_clip / pytorch360convert / DINOv2) | Medium | Phase 1 smoke gate before any wiring; WSL2 subprocess fallback documented |
| `erp_to_cubemap` signature / face-order mismatch → warped scene | Medium | Read the helper in full in Phase 1; reproject known-distance feature |
| Normals world-frame axis/sign ≠ converter frame → wrong grazing clip | Medium | Phase 3 axis check; additive fallback to depth-gradient path |
| CC BY-NC 4.0 blocks commercial deliverables | Certain (for now) | DA360/DAP stay defaults; CLI + README NC warning; resolve licensing later |
| Native depth detail capped at 504 px/face softens fine geometry | Medium-High | Upsample-before-projection keeps grid aligned; keep DAP for foliage-heavy scenes; log `native_resolution` |
| Vendored repo drifts / breaks on update | Low | Pin a commit; vendored copy, not live submodule |
| Zero-shot quality poor on natural landscapes | Medium | Phase 1 content A/B; keep DAP default if PaGeR underperforms |

---

## 9. Out of scope
- Fine-tuning to SPAG-4D render statistics (PanoInfinigen training data unreleased).
- Normal-neighbor floater pruning (optional future add).
- Metric-depth-only / normals-only single-task checkpoints (the unified checkpoint covers all heads).
- Any change to SPAG projection, PLY color-space paths, or the DA360/DAP/SHARP360 code.
