# SPAG-4D

Convert 360° panoramic photos into explorable 3D Gaussian Splat scenes.

SPAG-4D takes an equirectangular panorama, estimates depth with [DAP](https://github.com/Insta360-Research-Team/DAP) or [DA360](https://github.com/Insta360-Research-Team/DA360), and converts it into a 3D Gaussian Splat using spherical projection. For higher quality, optional per-face [ML-SHARP](https://github.com/apple/ml-sharp) refinement is available.

<p align="center">
  <img src="assets/demo.gif" alt="SPAG-4D demo — panorama to 3D Gaussian splat" width="720">
</p>

---

## Quick Start (Windows)

> Requires an NVIDIA GPU (6 GB+ VRAM), [Git](https://git-scm.com/downloads), and ~8 GB disk space. No Python or CUDA toolkit install needed -- everything is self-contained.

1. Download and extract the SPAG-4D release `.zip`.
2. Double-click **`install.bat`** and wait for "Installation Complete!"
3. Double-click **`run.bat`**.
4. Your browser opens to **http://localhost:7860** with a demo panorama loaded. Hit **Convert**.

See [INSTALL.md](INSTALL.md) for the full walkthrough and troubleshooting.

---

## How It Works

SPAG-4D supports two pipeline modes:

### SPAG Mode (Default -- Fast)

```
360° panorama
  -> Depth estimation (DAP or DA360)
  -> Spherical projection: depth * ray_direction = 3D positions
  -> Colors from panorama pixels (sRGB), latitude-aware scales
  -> Sky detection + pole thinning
  -> PLY export (~2s on GPU)
```

### SHARP Refined Mode (Optional -- Higher Quality)

```
360° panorama
  -> Depth estimation (DAP or DA360)
  -> Project onto cubemap/icosahedral faces
  -> Per-face SHARP inference (~1.18M Gaussians per face)
  -> Depth alignment + world-frame rotation + Voronoi merge
  -> PLY export (~60s+ on GPU)
```

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
# SPAG mode (fast, default)
python -m spag4d convert panorama.jpg output.ply

# Max quality SPAG (one Gaussian per pixel)
python -m spag4d convert panorama.jpg output.ply --stride 1

# Fast preview
python -m spag4d convert panorama.jpg output.ply --stride 4

# Use DA360 depth model instead of DAP
python -m spag4d convert panorama.jpg output.ply --depth-model da360

# SHARP refined mode (higher quality, slower)
python -m spag4d convert panorama.jpg output.ply --sharp-refine

# Batch convert a folder
python -m spag4d convert input_dir/ output_dir/ --batch

# Pre-download all model weights
python -m spag4d download-models
```

### Python API

```python
from spag4d import SPAG4D

# SPAG mode (fast)
converter = SPAG4D(device="cuda")
result = converter.convert("panorama.jpg", "output.ply", stride=2)

# SHARP refined mode
converter = SPAG4D(device="cuda", sharp_refine=True)
result = converter.convert("panorama.jpg", "output_sharp.ply")

# DA360 depth model
converter = SPAG4D(device="cuda", depth_model="da360")
result = converter.convert("panorama.jpg", "output_da360.ply")

print(f"{result.splat_count:,} Gaussians in {result.processing_time:.1f}s")
```

---

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `depth_model` | `dap` | Depth model: `dap` (metric depth) or `da360` (circular padding DPT) |
| `sharp_refine` | `false` | Enable SHARP per-face refinement (slower, higher quality) |
| `stride` | `2` | SPAG pixel stride: `1`=full density, `2`=quarter, `4`=sixteenth |
| `depth_min` | `0.1` | Clip geometry closer than this (meters) |
| `depth_max` | `100.0` | Clip geometry farther than this (meters) |
| `sky_threshold` | `80.0` | Depth cutoff for sky removal (0 = keep everything) |
| `outlier_pruning` | `0.0` | Statistical outlier removal (0 = off, 1 = aggressive) |
| `global_scale` | `1.0` | Multiply all depths by this factor |

### SPAG Stride Guide

| Stride | Gaussians (4096x2048 input) | File Size | Speed |
|--------|----------------------------|-----------|-------|
| 1 | ~1.3M | ~85 MB | ~2s |
| 2 | ~350K | ~23 MB | ~0.3s |
| 4 | ~85K | ~5.5 MB | ~0.1s |
| 8 | ~21K | ~1.4 MB | ~0.1s |

---

## Manual Setup (Linux / Mac / Developer)

```bash
git clone https://github.com/cedarconnor/SPAG4d.git
cd SPAG4d
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# DAP depth model architecture
git submodule update --init --recursive
# Or: git clone https://github.com/Insta360-Research-Team/DAP spag4d/dap_arch/DAP

# DA360 depth model architecture (optional)
git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360

# ML-SHARP (optional, only needed for --sharp-refine)
pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip

# Download model weights
python -m spag4d download-models
```

---

## Depth Models

| Model | Output | Best For |
|-------|--------|----------|
| **DAP** | Metric radial depth (meters) | General 360° scenes, consistent scale |
| **DA360** | Scale-invariant disparity | Circular-padding continuity, research comparison |

Both models download weights automatically on first use (~1.5 GB each).

---

## Project Structure

```
spag4d/
├── core.py                      # Pipeline orchestrator (SPAG + SHARP modes)
├── spag_converter.py            # SPAG: depth-to-Gaussian spherical projection
├── sharp_gaussian_pipeline.py   # SHARP: per-face inference + merge (optional)
├── dap_model.py                 # DAP depth estimation
├── da360_model.py               # DA360 depth estimation
├── projection.py                # Cubemap + icosahedral projectors
├── ply_writer.py                # PLY export (sRGB + linearRGB paths)
├── scene_filter.py              # Sky detection, pole thinning, outlier pruning
├── spherical_grid.py            # 360° coordinate math
└── cli.py                       # CLI commands

api.py                           # FastAPI web server
static/
├── index.html
├── css/style.css
└── js/
    ├── viewer.js                # GaussianSplats3D wrapper
    └── app.js                   # UI logic
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'spag4d.dap_arch.DAP.networks'` | `git submodule update --init --recursive` |
| CUDA out of memory | Use `--stride 4` or `--sharp-cubemap-size 768` |
| SHARP not found | `pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip` |
| DA360 not found | `git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360` |
| Port 7860 in use | Edit `run.bat` and change the port |

## References

- [DAP -- Depth Any Panorama](https://github.com/Insta360-Research-Team/DAP)
- [DA360 -- Depth Anything in 360](https://github.com/Insta360-Research-Team/DA360)
- [ML-SHARP -- Apple](https://github.com/apple/ml-sharp)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT (application code). SHARP model weights are under [Apple's license](https://github.com/apple/ml-sharp/blob/main/LICENSE).
