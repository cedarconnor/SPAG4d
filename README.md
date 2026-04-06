# SPAG-4D

Convert 360 panoramic photos into explorable 3D Gaussian Splat scenes.

SPAG-4D takes an equirectangular panorama, estimates depth with [DA360](https://github.com/Insta360-Research-Team/DA360) (default) or [DAP](https://github.com/Insta360-Research-Team/DAP), and converts it into a 3D Gaussian Splat using spherical projection. One Gaussian per pixel, colors taken directly from the source image, geometry from the depth model. Fast, accurate, no stitching artifacts.

Two refinement backends fill disocclusion holes from novel viewpoints:
- **GSFix3D** (default) -- scene-adapted diffusion inpainting per camera
- **OmniRoam v2** (optional) -- trajectory-coherent panoramic video generation via [OmniRoam](https://github.com/yuhengliu02/OmniRoam) with optional [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) video upscaling

<p align="center">
  <img src="assets/demo.gif" alt="SPAG-4D demo -- panorama to 3D Gaussian splat" width="720">
</p>

---

## Quick Start (Windows)

> Requires an NVIDIA GPU (6 GB+ VRAM for conversion, 16 GB+ for refinement, 48 GB for OmniRoam), [Git](https://git-scm.com/downloads), and ~30 GB disk space.

1. Download and extract the SPAG-4D release `.zip`.
2. Double-click **`install.bat`** and wait for "Installation Complete!"
3. Double-click **`run.bat`**.
4. Your browser opens to **http://localhost:7860** with a demo panorama loaded. Hit **Convert**.

See [INSTALL.md](INSTALL.md) for the full walkthrough and troubleshooting.

---

## How It Works

```
360 equirectangular panorama
  -> DA360 depth estimation (scale-invariant, circular-padding DPT)
  -> Scene analysis: auto-compute depth range, sky cutoff, orbit radius
  -> Spherical projection: depth * ray_direction = 3D Gaussian positions
  -> Colors sampled directly from source pixels (sRGB)
  -> Edge clipping, floater removal, sparse region pruning
  -> Standard PLY export
```

Each pixel in the panorama becomes one Gaussian splat (at stride=1) or every Nth pixel (at stride=2, 4, etc.). The depth model provides geometry, the source image provides color. No face stitching, no seam artifacts.

The output is a standard `.ply` file compatible with [SuperSplat](https://playcanvas.com/supersplat/editor), [gsplat](https://docs.gsplat.studio/), Blender, and any 3DGS viewer. SPAG-4D includes a built-in web viewer powered by [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D).

---

## Usage

### Web UI

```
run.bat
```

Or manually:

```
python -m spag4d serve --port 7860
```

Upload a 360 image, adjust settings, click **Convert**, and explore the result in the 3D viewer. Left-click to orbit, right-click to pan, scroll to zoom.

The refinement panel appears after conversion with a backend dropdown: **GSFix3D** (diffusion inpainting) or **OmniRoam v2** (trajectory-coherent video fill with optional SeedVR2 upscaling).

### Command Line

```bash
# Default (DA360 depth + SPAG conversion, auto scene defaults)
python -m spag4d convert panorama.jpg output.ply

# Max quality (one Gaussian per pixel)
python -m spag4d convert panorama.jpg output.ply --stride 1

# Fast preview
python -m spag4d convert panorama.jpg output.ply --stride 4

# Use DAP depth model instead of DA360
python -m spag4d convert panorama.jpg output.ply --depth-model dap

# Convert + fill disocclusion holes with GSFix3D
python -m spag4d convert panorama.jpg output.ply --refine

# Pre-download all model weights (including GSFix3D checkpoint)
python -m spag4d download-models
```

### Python API

```python
from spag4d import SPAG4D

converter = SPAG4D(device="cuda")

# Auto scene defaults (depth range, sky cutoff computed from depth map)
result = converter.convert("panorama.jpg", "output.ply", stride=2)

# Manual overrides
result = converter.convert("panorama.jpg", "output.ply",
    depth_min=0.5, depth_max=50.0, sky_threshold=30.0,
    grazing_angle=65.0)

print(f"{result.splat_count:,} Gaussians in {result.processing_time:.1f}s")
```

---

## Refinement

Single-viewpoint panoramas produce 3D Gaussians with structural holes -- areas behind foreground objects and at depth discontinuities that the original camera never observed. SPAG-4D offers two refinement backends to fill these holes.

### Backend 1: GSFix3D (Default)

Uses [GSFix3D](https://github.com/mobileroboticslab/GSFix3D) diffusion-guided novel view repair. Fast, runs entirely on Windows.

**How it works:**
1. **Camera rig** -- 36 novel-view cameras render the splat, detecting holes via alpha thresholding
2. **GSFixer inference** -- A scene-adapted diffusion model inpaints hole regions in each view
3. **Distillation** -- Repaired images are distilled back into the 3D Gaussians via differentiable rendering (L1 + SSIM loss)
4. **Iterate** -- Repeats until holes drop below 2%

**Requirements:** ~16 GB VRAM. GSFix3D checkpoint (~2 GB) downloads on first use.

```bash
python -m spag4d convert panorama.jpg output.ply --refine
```

### Backend 2: OmniRoam v2 (Optional)

Uses [OmniRoam](https://github.com/yuhengliu02/OmniRoam) to generate trajectory-coherent panoramic walkthrough video, then extracts perspective crops as pseudo-supervision for gap filling. Produces temporally consistent fill content instead of independent per-view inpainting.

**How it works:**
1. **Gap analysis** -- Render from 36 evaluation cameras, classify hole severity by direction (forward, left, right, backward)
2. **OmniRoam generation** -- Generate 81-frame 480x960 ERP video along gap-directed trajectories (runs in WSL2)
3. **SeedVR2 upscale** (optional) -- Upscale video from 480p to 1024p using [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) video restoration
4. **View selection** -- Extract perspective crops from frames that overlap gap regions, filter by gap ratio
5. **Gap seeding** -- Seed sparse Gaussians into gap regions using the source panorama's depth map
6. **Optimization** -- Distill with tier-1 (original cubemap, weight 1.0) + tier-2 (OmniRoam pseudo-views, weight 0.20)
7. **Validation** -- Source-anchor PSNR check, coverage measurement, PLY export

**Requirements:** WSL2 with Ubuntu, 48 GB VRAM (A6000 or better), ~20 GB disk for model weights.

#### OmniRoam Setup

```bash
# 1. Install OmniRoam in WSL2
wsl bash scripts/setup_omniroam_wsl.sh

# 2. (Optional) SeedVR2 is installed automatically by the setup script.
#    Model weights download on first use (~6 GB for 3B, ~14 GB for 7B).
```

#### OmniRoam Python API

```python
from spag4d.refine import refine_splat_v2
from spag4d.refine.omniroam_config import OmniRoamConfig

config = OmniRoamConfig(
    enabled=True,
    trajectory_mode="auto",         # "auto" | "all" | "forward" | ["forward", "left"]
    tier2_weight=0.20,              # OmniRoam pseudo-view loss weight
    upscale_backend="seedvr2",      # "none" | "seedvr2"
)

result = refine_splat_v2(
    ply_path="output.ply",
    panorama_path="panorama.jpg",
    depth_map=depth_array,          # (H, W) float32 from DA360/DAP
    config=config,
)
```

### Refinement Comparison

| | GSFix3D | OmniRoam v2 |
|---|---------|-------------|
| **Approach** | Per-view diffusion inpainting | Trajectory-coherent video generation |
| **Consistency** | Independent per camera | Temporally coherent across 81 frames |
| **Speed** | ~5 min | ~30 min (+ ~4 min with SeedVR2) |
| **VRAM** | 16 GB | 48 GB |
| **Platform** | Windows native | WSL2 (Linux) |
| **Upscaling** | N/A | Optional SeedVR2 (480p to 1024p) |

---

## Settings

### Conversion Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `depth_model` | `da360` | Depth model: `da360` (recommended) or `dap` (metric depth) |
| `stride` | `2` | Pixel stride: `1`=full density, `2`=quarter, `4`=sixteenth |
| `depth_min` | Auto | Clip geometry closer than this (meters). Auto = 1st percentile of depth. |
| `depth_max` | Auto | Clip geometry farther than this (meters). Auto = 99th percentile. |
| `sky_threshold` | Auto | Depth cutoff for sky removal. Auto = 95th percentile. |
| `grazing_angle` | `65` | Remove edge-on splats behind objects. 90=off, 65=default, 50=aggressive. |
| `outlier_pruning` | `0.3` | Floater removal strength. 0=off, 1=aggressive. |
| `sparse_pruning` | `0.3` | Remove isolated splats. 0=off, 1=aggressive. |
| `global_scale` | `1.0` | Multiply all depths by this factor |

Parameters marked "Auto" are computed from the depth map's statistical distribution, adapting to both indoor (3m rooms) and outdoor (100m forests) scenes without manual tuning. You can override any auto value by setting it explicitly.

### GSFix3D Refinement Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Cameras | 36 | Novel-view cameras for hole detection (12 directions x 3 depths) |
| Rounds | 3 | Maximum repair-distill cycles. Stops early when holes < 2% |
| Finetune Steps | 500 | GSFixer scene adaptation steps (~1 min per 100 steps) |

### OmniRoam v2 Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Trajectories | Auto | `auto` (gap-directed), `all` (4 cardinal), or specific presets |
| Rounds | 3 | Maximum refinement iterations |
| Tier-2 Weight | 0.20 | OmniRoam pseudo-view loss weight (0.05-0.50) |
| Upscale | None | `none` (480p) or `seedvr2` (1024p) |

### Stride Guide

| Stride | Gaussians (4096x2048 input) | File Size | Speed |
|--------|----------------------------|-----------|-------|
| 1 | ~5.6M | ~362 MB | ~3s |
| 2 | ~1.4M | ~90 MB | ~1s |
| 4 | ~350K | ~23 MB | ~0.3s |
| 8 | ~85K | ~5.5 MB | ~0.1s |

---

## Depth Models

| Model | Default | Description |
|-------|---------|-------------|
| **DA360** | Yes | Depth Anything V2 with circular-padding DPT decoder. Seamless 360 depth with no boundary artifacts. Superior results in most scenes. |
| **DAP** | No | Depth Any Panorama. Outputs metric radial depth. Alternative option. |

Both models download weights automatically on first use (~1.3-1.5 GB each).

---

## Manual Setup (Linux / Mac / Developer)

```bash
git clone https://github.com/cedarconnor/SPAG4d.git
cd SPAG4d
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# DA360 depth model (recommended)
git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360

# DAP depth model
git submodule update --init --recursive

# Download model weights
python -m spag4d download-models

# For GSFix3D refinement (optional, requires 16GB+ VRAM):
pip install diffusers transformers open3d trimesh scipy
python -m spag4d download-models --model gsfix3d

# For OmniRoam v2 refinement (optional, requires WSL2 + 48GB VRAM):
wsl bash scripts/setup_omniroam_wsl.sh
```

---

## Project Structure

```
spag4d/                          # Core conversion pipeline
  core.py                        # Pipeline orchestrator (auto scene defaults)
  scene_analysis.py              # Scale-relative parameter computation
  spag_converter.py              # Depth-to-Gaussian spherical projection
  dap_model.py                   # DAP depth estimation
  da360_model.py                 # DA360 depth estimation (default)
  ply_writer.py                  # PLY export (sRGB SH0 encoding)
  scene_filter.py                # Edge clipping, outlier pruning, sparse filtering
  spherical_grid.py              # 360 coordinate math
  cli.py                         # CLI commands

spag4d/refine/                   # Refinement pipelines
  pipeline.py                    # GSFix3D 3-phase refinement orchestrator
  pipeline_v2.py                 # OmniRoam 7-stage refinement orchestrator
  config.py                      # GSFix3D refinement hyperparameters
  omniroam_config.py             # OmniRoam + SeedVR2 configuration
  camera_rig.py                  # Novel-view camera generation + cubemap extraction
  gsfixer_adapter.py             # GSFixer diffusion model (fine-tune + inference)
  omniroam_adapter.py            # OmniRoam WSL2 subprocess wrapper
  omniroam_trajectory.py         # Trajectory generation (matches upstream OmniRoam)
  seedvr2_adapter.py             # SeedVR2 video upscale WSL2 wrapper
  gap_analysis.py                # Hole classification by angular direction
  view_selector.py               # Perspective crop extraction + gap-directed filtering
  scale_alignment.py             # Reprojection-based OmniRoam-to-splat scale alignment
  gap_seeding.py                 # Seed Gaussians into gaps from source depth
  validation.py                  # Source-anchor PSNR, coverage, multi-view agreement
  mesh_extract.py                # Poisson mesh for GSFix3D dual conditioning
  distill.py                     # Differentiable 3DGS optimization (L1 + SSIM + tier-2)
  format_compat.py               # PLY round-trip between SPAG-4D and GSFix3D
  provenance.py                  # Gaussian provenance tracking (original/densified/omniroam/gap_seed)

api.py                           # FastAPI web server + refine v1/v2 endpoints
static/
  index.html                     # Web UI with refinement backend toggle
  css/style.css
  js/
    viewer.js                    # GaussianSplats3D wrapper
    app.js                       # UI logic (conversion + dual-backend refinement)

scripts/
  setup_omniroam_wsl.sh          # WSL2 OmniRoam + SeedVR2 installation
  regenerate_trajectory_snapshots.sh  # Upstream trajectory parity verification
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'spag4d.dap_arch.DAP.networks'` | `git submodule update --init --recursive` |
| DA360 not found | `git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360` |
| CUDA out of memory (conversion) | Use `--stride 4` or lower input image resolution |
| CUDA out of memory (GSFix3D) | Needs ~16 GB VRAM. Reduce cameras or render resolution |
| CUDA out of memory (OmniRoam) | Needs ~48 GB VRAM. Use an A6000 or better |
| OmniRoam WSL2 not found | Run `wsl bash scripts/setup_omniroam_wsl.sh` |
| SeedVR2 BlockSwap error | Ensure `--dit_offload_device cpu` is set (handled automatically) |
| Port 7860 in use | Edit `run.bat` and change the port |
| Scene defaults look wrong | Override with explicit `depth_min`, `depth_max`, `sky_threshold` values |

## References

- [DA360 -- Depth Anything in 360](https://github.com/Insta360-Research-Team/DA360)
- [DAP -- Depth Any Panorama](https://github.com/Insta360-Research-Team/DAP)
- [GSFix3D -- Diffusion-Guided Novel View Repair](https://github.com/mobileroboticslab/GSFix3D)
- [OmniRoam -- Panoramic Video Generation](https://github.com/yuhengliu02/OmniRoam) (Adobe Research, SIGGRAPH 2026)
- [SeedVR2 -- Video Upscaling](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (ByteDance, ICLR 2026)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT. Note: OmniRoam is subject to Adobe Research License (noncommercial research only). SeedVR2 is MIT. The OmniRoam integration is an optional module -- core SPAG-4D remains MIT.
