# SPAG-4D

Convert 360° panoramic photos into explorable 3D Gaussian Splat scenes.

SPAG-4D takes an equirectangular panorama, estimates depth with [DA360](https://github.com/Insta360-Research-Team/DA360) (default) or [DAP](https://github.com/Insta360-Research-Team/DAP), and converts it into a 3D Gaussian Splat using spherical projection. One Gaussian per pixel, colors taken directly from the source image, geometry from the depth model. Fast, accurate, no stitching artifacts.

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

```
360° equirectangular panorama
  -> DA360 depth estimation (scale-invariant, circular-padding DPT)
  -> Spherical projection: depth * ray_direction = 3D Gaussian positions
  -> Colors sampled directly from source pixels (sRGB)
  -> Latitude-aware Gaussian scales, normal-aligned rotations
  -> Sky detection + pole thinning
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
# Default (DA360 depth + SPAG conversion)
python -m spag4d convert panorama.jpg output.ply

# Max quality (one Gaussian per pixel)
python -m spag4d convert panorama.jpg output.ply --stride 1

# Fast preview
python -m spag4d convert panorama.jpg output.ply --stride 4

# Use DAP depth model instead of DA360
python -m spag4d convert panorama.jpg output.ply --depth-model dap

# Batch convert a folder
python -m spag4d convert input_dir/ output_dir/ --batch

# Pre-download all model weights
python -m spag4d download-models
```

### Python API

```python
from spag4d import SPAG4D

converter = SPAG4D(device="cuda")
result = converter.convert("panorama.jpg", "output.ply", stride=2)

# DAP depth model
converter = SPAG4D(device="cuda", depth_model="dap")
result = converter.convert("panorama.jpg", "output_dap.ply")

print(f"{result.splat_count:,} Gaussians in {result.processing_time:.1f}s")
```

---

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `depth_model` | `da360` | Depth model: `da360` (recommended) or `dap` (metric depth) |
| `stride` | `2` | Pixel stride: `1`=full density, `2`=quarter, `4`=sixteenth |
| `depth_min` | `0.1` | Clip geometry closer than this (meters) |
| `depth_max` | `100.0` | Clip geometry farther than this (meters) |
| `sky_threshold` | `80.0` | Depth cutoff for sky removal (0 = keep everything) |
| `outlier_pruning` | `0.0` | Statistical outlier removal (0 = off, 1 = aggressive) |
| `global_scale` | `1.0` | Multiply all depths by this factor |

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
| **DA360** | Yes | Depth Anything V2 with circular-padding DPT decoder. Seamless 360° depth with no boundary artifacts. Produces superior results in most scenes. |
| **DAP** | No | Depth Any Panorama. Outputs metric radial depth. Alternative option. |

Both models download weights automatically on first use (~1.3-1.5 GB each).

> **Note on SHARP refinement:** An experimental `--sharp-refine` flag exists that runs Apple's [ML-SHARP](https://github.com/apple/ml-sharp) per-face neural inference. In practice, SHARP refinement produces lower quality than the default SPAG mode for panoramic scenes -- it generates fewer splats, introduces face-boundary artifacts, and takes 20x longer. It is retained for research/testing only.

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
# Or: git clone https://github.com/Insta360-Research-Team/DAP spag4d/dap_arch/DAP

# Download model weights
python -m spag4d download-models
```

---

## Project Structure

```
spag4d/
├── core.py                      # Pipeline orchestrator
├── spag_converter.py            # Depth-to-Gaussian spherical projection
├── dap_model.py                 # DAP depth estimation
├── da360_model.py               # DA360 depth estimation (default)
├── ply_writer.py                # PLY export (sRGB SH0 encoding)
├── scene_filter.py              # Sky detection, pole thinning, outlier pruning
├── spherical_grid.py            # 360° coordinate math
├── cli.py                       # CLI commands
├── sharp_gaussian_pipeline.py   # SHARP per-face inference (experimental)
└── projection.py                # Cubemap + icosahedral projectors (SHARP only)

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
| DA360 not found | `git clone https://github.com/Insta360-Research-Team/DA360 spag4d/da360_arch/DA360` |
| CUDA out of memory | Use `--stride 4` or lower input image resolution |
| Port 7860 in use | Edit `run.bat` and change the port |

## References

- [DA360 -- Depth Anything in 360](https://github.com/Insta360-Research-Team/DA360)
- [DAP -- Depth Any Panorama](https://github.com/Insta360-Research-Team/DAP)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT (application code). SHARP model weights are under [Apple's license](https://github.com/apple/ml-sharp/blob/main/LICENSE).
