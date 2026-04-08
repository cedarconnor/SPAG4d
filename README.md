# SPAG-4D

Convert 360 panoramic photos into explorable 3D Gaussian Splat scenes.

SPAG-4D takes an equirectangular panorama and converts it into a 3D Gaussian Splat using one of three generator backends. Two depth-based generators (DA360, DAP) project depth maps into Gaussians via spherical geometry. A third generator (SHARP 360) uses Apple's [SHARP](https://github.com/apple/ml-sharp) model to predict Gaussians directly from perspective face crops, with DA360 depth alignment for inter-face consistency.

Optional refinement fills disocclusion holes from novel viewpoints:
- **GSFix3D** (default) -- scene-adapted diffusion inpainting per camera
- **OmniRoam v2** (optional) -- trajectory-coherent panoramic video generation via [OmniRoam](https://github.com/yuhengliu02/OmniRoam) with optional [SeedVR2](https://github.com/TencentARC/SeedVR) video upscaling

Optional pre-processing upscales input with [SeedVR2](https://github.com/TencentARC/SeedVR) image upscaling (used natively on Windows for both SHARP face upscaling and OmniRoam video upscaling).

<p align="center">
  <img src="assets/demo.gif" alt="SPAG-4D demo -- panorama to 3D Gaussian splat" width="720">
</p>

---

## Quick Start (Windows)

> Requires an NVIDIA GPU (6 GB+ VRAM for DA360/DAP, 8 GB+ for SHARP 360, 16 GB+ for refinement), [Git](https://git-scm.com/downloads), and ~30 GB disk space.

1. Download and extract the SPAG-4D release `.zip`.
2. Double-click **`install.bat`** and wait for "Installation Complete!"
3. Double-click **`run.bat`**.
4. Your browser opens to **http://localhost:7860** with a demo panorama loaded. Select a generator and hit **Convert**.

See [INSTALL.md](INSTALL.md) for the full walkthrough and troubleshooting.

---

## Generators

SPAG-4D offers three ways to convert a panorama into Gaussians:

| Generator | How It Works | Speed | Quality | VRAM |
|-----------|-------------|-------|---------|------|
| **DA360** (default) | DA360 depth estimation + SPAG spherical projection | ~2s | Good | ~2 GB |
| **DAP** | DAP metric depth + SPAG spherical projection | ~3s | Good | ~3 GB |
| **SHARP 360** | Per-face SHARP prediction + DA360 alignment + merge | ~30s | Higher detail | ~8 GB |

### DA360 / DAP (Depth-Based)

```
360 equirectangular panorama
  -> Depth estimation (DA360 or DAP)
  -> Scene analysis: auto-compute depth range, sky cutoff, orbit radius
  -> Spherical projection: depth * ray_direction = 3D Gaussian positions
  -> Colors sampled directly from source pixels (sRGB)
  -> Edge clipping, floater removal, sparse region pruning
  -> Standard PLY export
```

Each pixel becomes one Gaussian (at stride=1) or every Nth pixel (at stride=2, 4, etc.). No face stitching, no seam artifacts.

### SHARP 360 (ML-Based)

```
360 equirectangular panorama
  -> Extract N perspective face crops (default: 6 horizon views with overlap)
  -> Optional SeedVR2 upscale per face
  -> Apple SHARP prediction per face (Gaussians predicted directly from image)
  -> Hard Voronoi border clipping (360/N degrees per face)
  -> DA360 depth alignment (smooth grid scale field for inter-face consistency)
  -> Rotate each face into world frame + merge
  -> Global scale restore + PLY export
```

SHARP predicts Gaussians directly from images using a learned model -- no separate depth estimation needed. The DA360 alignment step ensures faces agree on scale at their boundaries.

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

Upload a 360 image, select a generator (DA360, DAP, or SHARP 360), adjust settings, click **Convert**, and explore the result in the 3D viewer. Left-click to orbit, right-click to pan, scroll to zoom.

When SHARP 360 is selected, additional settings appear: side count (how many perspective views) and an optional SeedVR2 upscale checkbox.

The refinement panel appears after conversion with a backend dropdown: **GSFix3D** (diffusion inpainting) or **OmniRoam v2** (trajectory-coherent video fill with optional SeedVR2 upscaling).

### Command Line

```bash
# DA360 generator (default)
python -m spag4d convert panorama.jpg output.ply

# DAP generator
python -m spag4d convert panorama.jpg output.ply --generator dap

# SHARP 360 generator (6 faces, no upscale)
python -m spag4d convert panorama.jpg output.ply --generator sharp360

# SHARP 360 with 8 faces and SeedVR2 upscale
python -m spag4d convert panorama.jpg output.ply --generator sharp360 --side-count 8 --seedvr2-upscale

# DA360 with max quality (one Gaussian per pixel)
python -m spag4d convert panorama.jpg output.ply --stride 1

# DA360 fast preview
python -m spag4d convert panorama.jpg output.ply --stride 4

# Convert + fill disocclusion holes with GSFix3D
python -m spag4d convert panorama.jpg output.ply --refine

# Pre-download all model weights
python -m spag4d download-models
```

### Python API

```python
from spag4d import SPAG4D

converter = SPAG4D(device="cuda")

# DA360 generator (default)
result = converter.convert("panorama.jpg", "output.ply", stride=2)

# SHARP 360 generator
result = converter.convert("panorama.jpg", "output.ply",
    generator="sharp360", side_count=6)

# SHARP 360 with SeedVR2 face upscaling
result = converter.convert("panorama.jpg", "output.ply",
    generator="sharp360", side_count=8, seedvr2_upscale=True)

print(f"{result.splat_count:,} Gaussians in {result.processing_time:.1f}s")
```

---

## Refinement

Single-viewpoint panoramas produce 3D Gaussians with structural holes -- areas behind foreground objects and at depth discontinuities that the original camera never observed. SPAG-4D offers two refinement backends to fill these holes. Refinement works on any PLY regardless of which generator produced it.

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
1. **Gap analysis** -- Render from 36 evaluation cameras, classify hole severity by direction
2. **OmniRoam generation** -- Generate 81-frame 480x960 ERP video along gap-directed trajectories (runs in WSL2)
3. **SeedVR2 upscale** (optional) -- Upscale video from 480p to 1024p using SeedVR2 (runs natively on Windows)
4. **View selection** -- Extract perspective crops from frames that overlap gap regions
5. **Gap seeding** -- Seed sparse Gaussians into gap regions using the source panorama's depth map
6. **Optimization** -- Distill with tier-1 (original cubemap, weight 1.0) + tier-2 (OmniRoam pseudo-views, weight 0.20)
7. **Validation** -- Source-anchor PSNR check, coverage measurement, PLY export

**Requirements:** WSL2 with Ubuntu for OmniRoam generation, 48 GB VRAM (A6000 or better), ~20 GB disk for model weights.

#### OmniRoam Setup

```bash
# 1. Install OmniRoam in WSL2
wsl bash scripts/setup_omniroam_wsl.sh

# 2. SeedVR2 runs natively on Windows (no WSL2 needed for upscaling)
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
| **Platform** | Windows native | WSL2 (OmniRoam) + Windows (SeedVR2 upscale) |
| **Upscaling** | N/A | Optional SeedVR2 (480p to 1024p, native Windows) |

---

## SeedVR2 Upscaling

[SeedVR2](https://github.com/TencentARC/SeedVR) provides neural image and video upscaling. SPAG-4D uses it natively on Windows in two contexts:

| Context | Mode | When |
|---------|------|------|
| **SHARP 360 face upscale** | Image | Before SHARP prediction -- upscale each perspective face crop |
| **OmniRoam video upscale** | Video | After OmniRoam generation -- upscale 480p video to 1024p |

### Setup

```bash
# Clone SeedVR2 into third_party/
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git third_party/seedvr2_videoupscaler

# Download model weights (~6.4 GB for 3B model)
# Place in third_party/seedvr2_videoupscaler/models/seedvr2_ema_3b_fp16.safetensors
```

SeedVR2 runs as a native Windows subprocess -- no WSL2 required.

---

## Settings

### Generator Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `generator` | `da360` | Generator backend: `da360`, `dap`, or `sharp360` |

### DA360/DAP Settings (Depth-Based)

| Setting | Default | Description |
|---------|---------|-------------|
| `stride` | `2` | Pixel stride: `1`=full density, `2`=quarter, `4`=sixteenth |
| `depth_min` | Auto | Clip geometry closer than this (meters). Auto = 1st percentile of depth. |
| `depth_max` | Auto | Clip geometry farther than this (meters). Auto = 99th percentile. |
| `sky_threshold` | Auto | Depth cutoff for sky removal. Auto = 95th percentile. |
| `grazing_angle` | `65` | Remove edge-on splats behind objects. 90=off, 65=default, 50=aggressive. |
| `outlier_pruning` | `0.3` | Floater removal strength. 0=off, 1=aggressive. |
| `sparse_pruning` | `0.3` | Remove isolated splats. 0=off, 1=aggressive. |
| `global_scale` | `1.0` | Multiply all depths by this factor |

### SHARP 360 Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `side_count` | `6` | Number of horizon perspective views (4, 6, 8, 10, or 12) |
| `seedvr2_upscale` | `false` | Upscale face images with SeedVR2 before SHARP prediction |

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
| Upscale | None | `none` (480p) or `seedvr2` (1024p, native Windows) |

### Stride Guide (DA360/DAP Only)

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
| **DA360** | Yes | Depth Anything V2 with circular-padding DPT decoder. Seamless 360 depth with no boundary artifacts. |
| **DAP** | No | Depth Any Panorama. Outputs metric radial depth. |

Both models download weights automatically on first use (~1.3-1.5 GB each). DA360 is also used internally by SHARP 360 for inter-face depth alignment.

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

# For SHARP 360 generator (optional):
pip install plyfile
# ml-sharp is already vendored in spag4d/sharp_arch/ml-sharp/
# SHARP checkpoint auto-downloads on first use (~500 MB)

# For SeedVR2 upscaling (optional):
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git third_party/seedvr2_videoupscaler
# Download model weights into third_party/seedvr2_videoupscaler/models/

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
  core.py                        # Pipeline orchestrator (generator dispatch)
  sharp360.py                    # SHARP 360 generator (face extraction + prediction + merge)
  seedvr2.py                     # Native Windows SeedVR2 adapter (image + video)
  scene_analysis.py              # Scale-relative parameter computation
  spag_converter.py              # Depth-to-Gaussian spherical projection
  dap_model.py                   # DAP depth estimation
  da360_model.py                 # DA360 depth estimation (default)
  ply_writer.py                  # PLY export (sRGB SH0 encoding)
  scene_filter.py                # Edge clipping, outlier pruning, sparse filtering
  spherical_grid.py              # 360 coordinate math
  cli.py                         # CLI commands

spag4d/sharp_arch/ml-sharp/      # Vendored Apple SHARP model
  src/sharp/models/              # SHARP predictor architecture
  src/sharp/utils/               # Gaussians3D, color space, PLY export
  src/sharp/cli/predict.py       # predict_image() inference function

spag4d/refine/                   # Refinement pipelines
  pipeline.py                    # GSFix3D 3-phase refinement orchestrator
  pipeline_v2.py                 # OmniRoam 7-stage refinement orchestrator
  config.py                      # GSFix3D refinement hyperparameters
  omniroam_config.py             # OmniRoam + SeedVR2 configuration
  camera_rig.py                  # Novel-view camera generation + cubemap extraction
  gsfixer_adapter.py             # GSFixer diffusion model (fine-tune + inference)
  omniroam_adapter.py            # OmniRoam WSL2 subprocess wrapper
  omniroam_trajectory.py         # Trajectory generation (matches upstream OmniRoam)
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
  index.html                     # Web UI with generator toggle + refinement backend
  css/style.css
  js/
    viewer.js                    # GaussianSplats3D wrapper
    app.js                       # UI logic (generator selection + refinement)

scripts/
  setup_omniroam_wsl.sh          # WSL2 OmniRoam installation
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'spag4d.dap_arch.DAP.networks'` | `git submodule update --init --recursive` |
| DA360 not found | `git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360` |
| SHARP checkpoint download fails | Check network connection; checkpoint is ~500 MB from `ml-site.cdn-apple.com` |
| `No module named 'sharp'` | Ensure `spag4d/sharp_arch/ml-sharp/src/` exists (should be vendored) |
| `No module named 'plyfile'` | `pip install plyfile` (required for SHARP PLY export) |
| SeedVR2 not found | Clone into `third_party/seedvr2_videoupscaler/` and download model weights |
| CUDA out of memory (conversion) | Use DA360 with `--stride 4`, or reduce panorama resolution |
| CUDA out of memory (SHARP 360) | Reduce `--side-count 4`, or disable `--seedvr2-upscale` |
| CUDA out of memory (GSFix3D) | Needs ~16 GB VRAM. Reduce cameras or render resolution |
| CUDA out of memory (OmniRoam) | Needs ~48 GB VRAM. Use an A6000 or better |
| OmniRoam WSL2 not found | Run `wsl bash scripts/setup_omniroam_wsl.sh` |
| Port 7860 in use | Edit `run.bat` and change the port |
| Scene defaults look wrong | Override with explicit `depth_min`, `depth_max`, `sky_threshold` values |

## References

- [SHARP -- Single-image to 3D Gaussians](https://github.com/apple/ml-sharp) (Apple, CVPR 2025)
- [SHARP 360 to Splat](https://github.com/Enndee/SHARP_360_to_Splat) -- reference implementation for 360 SHARP pipeline
- [DA360 -- Depth Anything in 360](https://github.com/Insta360-Research-Team/DA360)
- [DAP -- Depth Any Panorama](https://github.com/Insta360-Research-Team/DAP)
- [GSFix3D -- Diffusion-Guided Novel View Repair](https://github.com/mobileroboticslab/GSFix3D)
- [OmniRoam -- Panoramic Video Generation](https://github.com/yuhengliu02/OmniRoam) (Adobe Research, SIGGRAPH 2026)
- [SeedVR2 -- Video/Image Upscaling](https://github.com/TencentARC/SeedVR) (ByteDance, ICLR 2026)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT. Note: SHARP model weights are subject to Apple's model license (noncommercial research only). SHARP source code is Apple MIT-equivalent. OmniRoam is subject to Adobe Research License (noncommercial research only). SeedVR2 is MIT. These integrations are optional modules -- core SPAG-4D remains MIT.
