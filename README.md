# SPAG-4D

Convert 360° panoramic photos into explorable 3D Gaussian Splat scenes.

SPAG-4D projects your equirectangular panorama onto cubemap or icosahedral faces, runs Apple's [ML-SHARP](https://github.com/apple/ml-sharp) on each face to predict dense 3D Gaussians, aligns them to metric depth from [DAP](https://github.com/Insta360-Research-Team/DAP), and merges everything into a standard PLY file you can view in any Gaussian Splatting renderer.

<div align="center">

https://github.com/cedarconnor/SPAG4d/raw/main/assets/demo.mp4

</div>

---

## Quick Start (Windows)

> Requires an NVIDIA GPU (8 GB+ VRAM), [Git](https://git-scm.com/downloads), and ~6 GB disk space. No Python or CUDA toolkit install needed -- everything is self-contained.

1. Download and extract the SPAG-4D release `.zip`.
2. Double-click **`install.bat`** and wait for "Installation Complete!"
3. Double-click **`run.bat`**.
4. Your browser opens to **http://localhost:7860** with a demo panorama loaded. Hit **Convert**.

See [INSTALL.md](INSTALL.md) for the full walkthrough and troubleshooting.

---

## How It Works

```
360° equirectangular image
  │
  ├─ DAP depth estimation (metric radial depth for the full sphere)
  │
  ├─ Project onto faces (cubemap: 6 faces, icosahedral: 20 faces)
  │
  ├─ Per-face SHARP inference (predicts ~1.18M Gaussians per face)
  │      positions, scales, rotations, colors, opacities
  │
  ├─ DAP alignment (log-median scale matching per face)
  │
  ├─ World-frame rotation + Voronoi ownership merge
  │
  ├─ Seam color smoothing + optional outlier pruning
  │
  └─ PLY export (linearRGB → sRGB, standard 3DGS format)
```

The output is a standard `.ply` file compatible with [SuperSplat](https://playcanvas.com/supersplat/editor), [gsplat](https://docs.gsplat.studio/), Blender, and any 3DGS viewer. SPAG-4D also includes a built-in web viewer powered by [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D).

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
# Basic conversion
python -m spag4d convert panorama.jpg output.ply

# Higher quality with icosahedral projection (20 faces instead of 6)
python -m spag4d convert panorama.jpg output.ply --sharp-projection icosahedral

# Lower VRAM usage
python -m spag4d convert panorama.jpg output.ply --sharp-cubemap-size 768

# Batch convert a folder
python -m spag4d convert input_dir/ output_dir/ --batch

# Pre-download model weights (~4.5 GB)
python -m spag4d download-models
```

### Python API

```python
from spag4d import SPAG4D

converter = SPAG4D(device="cuda")
result = converter.convert("panorama.jpg", "output.ply")

print(f"{result.splat_count:,} Gaussians in {result.processing_time:.1f}s")
```

---

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `sharp_projection` | `cubemap` | `cubemap` (6 faces, fast) or `icosahedral` (20 faces, better pole coverage) |
| `sharp_cubemap_size` | `1536` | Resolution per face. Higher = more detail, more VRAM |
| `depth_min` | `0.1` | Clip geometry closer than this (meters) |
| `depth_max` | `100.0` | Clip geometry farther than this (meters) |
| `sky_threshold` | `80.0` | Depth cutoff for sky removal (0 = keep everything) |
| `grid_jitter` | `0.03` | Anti-aliasing jitter (0 = off, 0.5 = max) |
| `outlier_pruning` | `0.0` | Statistical outlier removal (0 = off, 1 = aggressive) |
| `global_scale` | `1.0` | Multiply all depths by this factor |

### VRAM Requirements

| Face Size | Quality | VRAM |
|-----------|---------|------|
| 768 | Low | ~3 GB |
| 1536 | Default | ~6 GB |
| 1920 | High | ~8 GB |
| 2304 | Very High | ~12 GB |
| 3072 | Ultra | ~16 GB+ |

---

## Manual Setup (Linux / Mac / Developer)

```bash
git clone --recurse-submodules https://github.com/cedarconnor/SPAG4d.git
cd SPAG4d
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[server,download]"
pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip
```

If submodules weren't cloned, run `git submodule update --init --recursive` for DAP.

---

## Project Structure

```
spag4d/
├── core.py                      # Pipeline orchestrator
├── sharp_gaussian_pipeline.py   # SHARP inference + face merging
├── dap_model.py                 # DAP depth estimation
├── projection.py                # Cubemap + icosahedral projectors
├── ply_writer.py                # PLY export (linearRGB→sRGB + SH0)
├── scene_filter.py              # Sky detection, outlier pruning
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
| CUDA out of memory | `--sharp-cubemap-size 768` or use `icosahedral` projection |
| SHARP not found | `pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip` |
| Port 7860 in use | Edit `run.bat` and change the port |

## References

- [DAP -- Depth Any Panorama](https://github.com/Insta360-Research-Team/DAP)
- [ML-SHARP -- Apple](https://github.com/apple/ml-sharp)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT (application code). SHARP model weights are under [Apple's license](https://github.com/apple/ml-sharp/blob/main/LICENSE).
