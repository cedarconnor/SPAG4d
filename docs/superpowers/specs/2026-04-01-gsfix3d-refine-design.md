# GSFix3D-Based Disocclusion Repair — Implementation Spec

**Date:** 2026-04-01
**Status:** Approved
**Replaces:** Klein/FLUX-based refinement pipeline (`spag4d-refine/`)

---

## Summary

Replace the non-functional Klein-based refinement system with GSFix3D, a diffusion-guided approach that repairs disocclusion holes via render-repair-distill rather than 2D-inpaint-then-backproject. The fundamental shift: depth placement becomes an optimization variable solved by differentiable rendering, not an input that must be correct up-front.

**Key decisions made during brainstorming:**
- Clean-room replacement (new `spag4d/refine/`, delete `spag4d-refine/` entirely)
- GSFix3D as git submodule in `third_party/GSFix3D/`
- Scaffold-first implementation (stubs → wire UI → flesh out internals)
- Full end-to-end UI: upload → convert → refine → view refined PLY
- Target hardware: NVIDIA A6000 (48GB VRAM), CUDA, Windows 11

---

## 1. Module Structure

### New: `spag4d/refine/`

```
spag4d/refine/
├── __init__.py           # Public API: refine_splat()
├── pipeline.py           # Top-level orchestrator (iterative repair loop)
├── config.py             # RefineConfig dataclass (all hyperparameters)
├── camera_rig.py         # Phase 1: camera placement, rendering, hole detection
├── mesh_extract.py       # Mesh extraction for GSFixer dual conditioning
├── gsfixer_adapter.py    # Phase 2: GSFixer load / finetune / inference
├── distill.py            # Phase 3: 3DGS optimization via differentiable rendering
├── provenance.py         # Tag Gaussians as original vs. new (LR scaling)
└── format_compat.py      # PLY format conversion (SPAG-4D <-> GSFix3D)
```

### External: `third_party/GSFix3D/` (git submodule)

- Pinned to specific commit
- `diff-gaussian-rasterization` compiled into `.venv`
- Imported via `sys.path` insertion (same pattern as DAP)

### New pip dependencies

- `trimesh` — mesh extraction
- `open3d` — Poisson surface reconstruction
- `diff-gaussian-rasterization` — CUDA C++ extension from GSFix3D submodule

### Unchanged

- `spag4d/core.py`, `spag4d/spag_converter.py`, `spag4d/ply_writer.py` — untouched
- The existing convert flow (upload -> depth -> PLY) is fully preserved

---

## 2. Pipeline Architecture

### Three-phase iterative loop

```
Raw PLY (from SPAG-4D core)
  │
  ├── Phase 1: Camera Rig & Hole Detection
  │     Generate 36 novel-view cameras (12 azimuths x 3 depths)
  │     Translate away from origin to expose parallax holes
  │     Render each view, extract binary hole masks (alpha < 0.1)
  │     Filter to cameras with >3% hole pixels (up to 20)
  │
  ├── Phase 2: GSFixer Fine-Tune & Inference
  │     Extract conditioning mesh from DAP depth (Poisson + simplify 10%)
  │     Fine-tune GSFixer on scene (6 cubemap GT views, 500 steps, ~5 min)
  │     Run inference on hole-containing renders (dual conditioning: mesh + GS)
  │     Output: repaired images with holes filled
  │
  └── Phase 3: 3DGS Distillation
        Optimize Gaussians so renders match repaired images
        L1 + D-SSIM loss, weighted higher in hole regions
        Mixed training: 70% repaired views, 30% original cubemap GTs
        Adaptive densification every 100 steps (3000 total)
        Original Gaussians get 0.1x LR to prevent drift

Iterate up to 3 times until avg hole fraction < 2%
```

### Camera placement strategy

Panorama splats have dense angular coverage from origin but zero translational baseline. Holes only appear with camera translation.

- 12 azimuthal directions (30 deg apart)
- 3 translation distances per direction: 5%, 15%, 30% of median scene depth
- Each camera: 512x512 perspective, 60 deg FOV, looking back toward origin
- Total: 36 cameras, filtered down to ~20 with significant holes

### GSFixer adaptation for panorama input

GSFix3D was trained on SLAM reconstructions (multi-camera). Adaptations:
- Camera rig explicitly generates the translational diversity SLAM provides naturally
- Scene-specific fine-tuning (500 steps) bridges the distribution gap
- Mesh conditioning from DAP depth gives structural context even where rough

### Convergence

- Stop when avg hole fraction across all cameras < 2%
- Or after max 3 iterations
- VRAM freed between phases (sequential, not concurrent)
- Peak: ~24GB during Phase 2 fine-tuning (comfortable on A6000)

---

## 3. API Changes

### `POST /api/refine` — simplified

**Removed parameters:** `synthesis_backend`, `camera_preset`, `custom_cameras`, `orbit_radius`
**Kept:** `job_id`, `max_rounds`
**Added:** `num_cameras` (default 36), `finetune_steps` (default 500)

Backend is always GSFix3D — no selector needed.

### `_run_refinement()` — rewired

Calls `spag4d.refine.pipeline.refine_splat()` directly. No `sys.path` hack, no separate package import.

### Progress stages (new)

```
camera_rig        → "Generating cameras"
mesh_extract      → "Extracting mesh"
finetune          → "Adapting to scene"
render_holes      → "Detecting holes"
gsfixer_inference → "Repairing holes"
distill           → "Optimizing 3D"
```

### Endpoints kept as-is

- `GET /api/refine/status/{id}` — same shape, new stage names
- `GET /api/refine/download/{id}` — unchanged
- `GET /api/refine/diagnostics/{id}` — unchanged format

### Endpoints removed

- `GET /api/refine/heatmap/{id}` — skip for now

---

## 4. UI Changes

### Refine panel — simplified

**Removed:** backend selector, orbit radius slider, camera preset dropdown
**Kept:** Refine button, max rounds control
**Added:** num cameras slider, finetune steps (collapsed "Advanced" section)

### Progress display

Stage labels mapped to human-readable names (see Section 3). Same progress bar, same polling pattern (`/api/refine/status/{id}` every 2 seconds).

### Result display

On completion: load refined PLY into viewer, show metrics (holes before/after, Gaussian count, time), download button. Same flow as current, just with different metric names.

### Unchanged

Upload flow, convert flow, 3D viewer, tab layout, styling, depth preview.

---

## 5. Klein Removal Scope

### Deleted entirely

- `spag4d-refine/` — entire directory
  - All submodules: synthesis/, camera/, gaussian/, optimization/, regions/, renderer/, seeding/, validation/
  - CLI, pipeline, config, session, tests, pyproject.toml

### Cleaned in `api.py`

- Remove `sys.path` hack for `spag4d-refine`
- Remove old imports (`spag4d_refine.config`, `spag4d_refine.pipeline`)
- Remove Klein-specific parameters
- Remove heatmap endpoint

### Cleaned in UI

- Remove backend selector, orbit radius, camera preset controls
- Update stage name mappings

---

## 6. Implementation Waves

### Wave 1: Scaffold (stubs + wiring)
1. Add GSFix3D submodule, build diff-gaussian-rasterization
2. Create `spag4d/refine/` with stub implementations returning fake data
3. Rewire `api.py` to new module
4. Update UI (simplify refine panel, update stage labels)
5. Test through Chrome: full stub flow works end-to-end

### Wave 2: Phase 1 (camera rig + hole detection)
6. Implement camera_rig.py (real camera placement, rendering, masks)
7. Implement mesh_extract.py (depth-to-mesh)
8. Test: cameras generate, renders show actual holes

### Wave 3: Phase 2 (GSFixer)
9. Implement gsfixer_adapter.py (load, finetune, inference)
10. Implement format_compat.py (PLY round-trip)
11. Add checkpoint download to CLI
12. Test: fine-tuning runs, inference produces repaired images

### Wave 4: Phase 3 (distillation)
13. Implement distill.py (optimization loop with densification)
14. Implement provenance.py (Gaussian tagging, LR scaling)
15. Test: optimization reduces loss, holes decrease

### Wave 5: Integration & cleanup
16. End-to-end test on real panorama through UI
17. Delete `spag4d-refine/`
18. Clean api.py dead code
19. Final UI test via Chrome

### Wave 6: CLI
20. Add `--refine` flag to convert command
21. Add GSFix3D checkpoint to download-models

---

## 7. Risk Mitigations

### Format compatibility
Write round-trip test early (Wave 2): load SPAG-4D PLY into GSFix3D format, save back, verify renders match. Both use standard 3DGS PLY (WXYZ quaternions, logit opacity, log scale).

### Stable Diffusion v2 availability
SD v2 removed from HuggingFace. Check GSFix3D pretrained checkpoints for bundled weights. If not included, source from third-party mirror.

### Panorama adaptation gap
Start with obvious test case (furniture against wall). Camera rig parameters (`translation_fracs`, `num_directions`) are tunable if initial results are poor.

### VRAM management
Explicitly `del` depth model and `torch.cuda.empty_cache()` before entering refine pipeline. Peak ~24GB during fine-tuning, well within A6000's 48GB.

---

## 8. Configuration Defaults

```python
@dataclass
class RefineConfig:
    # Paths
    gsfixer_checkpoint: str = "pretrained/gsfix3d/gsfix3d_base.ckpt"

    # Phase 1: Camera Rig
    camera_fov: float = 60.0
    translation_fracs: tuple = (0.05, 0.15, 0.30)
    alpha_threshold: float = 0.1
    min_hole_fraction: float = 0.03
    max_repair_cameras: int = 20

    # Phase 2: GSFixer
    finetune_steps: int = 500
    finetune_lr: float = 1.0e-5
    inference_steps: int = 50
    guidance_scale: float = 7.5
    mesh_simplify_ratio: float = 0.1

    # Phase 3: Distillation
    distill_iterations: int = 3000
    densify_grad_threshold: float = 0.0002
    lr_position: float = 0.00016
    lr_feature: float = 0.0025
    lr_opacity: float = 0.05
    lr_scaling: float = 0.005
    lr_rotation: float = 0.001
    original_view_ratio: float = 0.3

    # Convergence
    convergence_threshold: float = 0.02
    max_iterations: int = 3
```
