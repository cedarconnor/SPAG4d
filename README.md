# SPAG-4D

Convert 360° panoramic photos into explorable 3D Gaussian Splat scenes.

SPAG-4D takes an equirectangular panorama, estimates depth with [DA360](https://github.com/Insta360-Research-Team/DA360) (default) or [DAP](https://github.com/Insta360-Research-Team/DAP), and converts it into a 3D Gaussian Splat using spherical projection. One Gaussian per pixel, colors taken directly from the source image, geometry from the depth model. Fast, accurate, no stitching artifacts.

Includes an AI-powered refinement pipeline using [Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b) + [ml-sharp LoRA](https://huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair) to detect and fill gaps with 3D-consistent content, followed by differentiable optimization with DINOv2 semantic and Pearson depth losses.

<p align="center">
  <img src="assets/demo.gif" alt="SPAG-4D demo — panorama to 3D Gaussian splat" width="720">
</p>

---

## Quick Start (Windows)

> Requires an NVIDIA GPU (6 GB+ VRAM for conversion, 24 GB+ for refinement with Klein), [Git](https://git-scm.com/downloads), and ~30 GB disk space.

1. Download and extract the SPAG-4D release `.zip`.
2. Double-click **`install.bat`** and wait for "Installation Complete!"
3. Double-click **`run.bat`**.
4. Your browser opens to **http://localhost:7860** with a demo panorama loaded. Hit **Convert**.

See [INSTALL.md](INSTALL.md) for the full walkthrough and troubleshooting.

---

## How It Works

```
360° equirectangular panorama
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

Upload a 360° image, adjust settings, click **Convert**, and explore the result in the 3D viewer. Left-click to orbit, right-click to pan, scroll to zoom.

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

# Pre-download all model weights
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

## Splat Refinement

After conversion, SPAG-4D can refine the Gaussian splat by detecting and filling gaps using AI synthesis. The refinement pipeline:

1. **Analyzes** the splat from candidate viewpoints to find where gaps exist (gap-driven camera selection)
2. **Renders** the splat from selected cameras and compares against parallax-corrected panoramic projections
3. **Classifies** each pixel as Trusted, Degraded (TYPE_A), or Gap (TYPE_C)
4. **Synthesizes** repaired views using Klein 9B + ml-sharp LoRA with 6DoF camera-aware prompting
5. **Seeds** new Gaussians in gap regions using monocular depth estimation + affine alignment
6. **Validates** new Gaussians via multi-view visibility and color consistency checks
7. **Optimizes** seeded Gaussians using differentiable gsplat rendering with masked L1, DINOv2 semantic, and Pearson depth losses
8. **Prunes** low-quality and inconsistent Gaussians

### Camera Modes

| Mode | Description |
|------|-------------|
| **Auto (gap-driven)** | Renders the splat from 14 candidate viewpoints, selects cameras that see the most gaps. Default and recommended. |
| **Orbit** | Fixed horizontal camera ring at scene center |
| **Orbit + Above/Below** | Horizontal + elevated/lowered cameras |
| **Full Sphere** | Three rings (0°, +30°, -20°) for maximum coverage |
| **Custom** | Navigate the 3D viewer and click "Capture Viewpoint" to place cameras manually |

### Provenance Heatmap

After refinement, click the **Heatmap** button to color-code Gaussians by origin:

- **Blue** = Original (from initial conversion)
- **Green** = Promoted (new, validated across multiple views)
- **Yellow** = Seeded (new, candidate for promotion)

### Refinement Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Camera Mode | Auto | Camera placement strategy (Auto analyzes splat for gaps) |
| Radius | Auto | Camera distance from center (auto-computed from scene depth) |
| Cameras | 8 | Number of viewpoints (~5 min each with Klein 9B) |
| Rounds | 1 | Refinement iterations (1 is usually sufficient) |

> **Requirements:** Refinement needs `diffusers>=0.37.0`, `transformers`, `gsplat>=1.5`, and ~24 GB VRAM. Klein 9B model weights (~18 GB) download automatically on first use. The ml-sharp LoRA (~260 MB) downloads from Hugging Face.

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
| **DA360** | Yes | Depth Anything V2 with circular-padding DPT decoder. Seamless 360° depth with no boundary artifacts. Superior results in most scenes. |
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

# For refinement (optional, requires 24GB+ VRAM):
pip install -r requirements-refine.txt
```

---

## Project Structure

```
spag4d/                          # Core conversion pipeline
├── core.py                      # Pipeline orchestrator (auto scene defaults)
├── scene_analysis.py            # Scale-relative parameter computation
├── spag_converter.py            # Depth-to-Gaussian spherical projection
├── dap_model.py                 # DAP depth estimation
├── da360_model.py               # DA360 depth estimation (default)
├── ply_writer.py                # PLY export (sRGB SH0 encoding)
├── scene_filter.py              # Edge clipping, outlier pruning, sparse filtering
├── spherical_grid.py            # 360° coordinate math
└── cli.py                       # CLI commands

spag4d-refine/                   # Refinement pipeline (gap filling)
├── spag4d_refine/
│   ├── pipeline.py              # Multi-stage refinement orchestrator
│   ├── config.py                # All refinement parameters
│   ├── camera/                  # Gap-driven camera selection, panoramic extraction
│   ├── gaussian/                # GaussianCloud with provenance tracking
│   ├── regions/                 # Three-way gap region classification
│   ├── renderer/                # gsplat rendering + diagnostics
│   ├── seeding/                 # Shadow Gaussian creation + multi-view validation
│   ├── synthesis/               # Klein 9B + ml-sharp LoRA (with fused QKV splitting)
│   ├── optimization/            # Masked gsplat optimization (L1 + DINOv2 + depth)
│   └── validation/              # PSNR checks, pruning, metrics
└── tests/

api.py                           # FastAPI web server + refine endpoints
static/
├── index.html                   # Web UI with inline help
├── css/style.css
└── js/
    ├── viewer.js                # GaussianSplats3D wrapper
    └── app.js                   # UI logic (conversion + refinement)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'spag4d.dap_arch.DAP.networks'` | `git submodule update --init --recursive` |
| DA360 not found | `git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360` |
| CUDA out of memory (conversion) | Use `--stride 4` or lower input image resolution |
| CUDA out of memory (refinement) | Klein 9B needs ~24 GB VRAM. Reduce cameras or use `--synthesis-backend sdxl` |
| gsplat LNK1104 error on Windows | Delete `%LOCALAPPDATA%\torch_extensions\...\gsplat_cuda` and restart |
| Port 7860 in use | Edit `run.bat` and change the port |
| Scene defaults look wrong | Override with explicit `depth_min`, `depth_max`, `sky_threshold` values |

## References

- [DA360 -- Depth Anything in 360](https://github.com/Insta360-Research-Team/DA360)
- [DAP -- Depth Any Panorama](https://github.com/Insta360-Research-Team/DAP)
- [Klein 9B -- FLUX.2 Image Editing](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b)
- [ml-sharp 3D Repair LoRA](https://huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair)
- [DINOv2 -- Self-supervised Vision Transformer](https://github.com/facebookresearch/dinov2)
- [gsplat -- Gaussian Splatting Library](https://github.com/nerfstudio-project/gsplat)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT
