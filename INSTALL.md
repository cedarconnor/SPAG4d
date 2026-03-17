# SPAG-4D Installation Guide

Step-by-step instructions for the Windows portable install. For Linux/Mac or pip-based setup, see the [README](README.md#manual-setup-linux--mac--developer).

---

## What You Need

- **Windows 10 or 11** (64-bit)
- **NVIDIA GPU** with 6 GB+ VRAM
- **NVIDIA Driver** 525 or newer
- **Git** -- download from [git-scm.com](https://git-scm.com/downloads) and make sure it's on your PATH
- **Internet connection** for the first-time download (~8 GB total)

You do **not** need to install Python, CUDA, or any other toolkits. The installer bundles everything.

---

## Step 1: Install

1. Download the SPAG-4D `.zip` release and extract it to a permanent location (e.g. `C:\SPAG-4D` or your Desktop).

2. Open the folder and double-click **`install.bat`**.
   - If Windows SmartScreen blocks it: click **More info** then **Run anyway**.

3. A terminal window opens and walks through the setup:
   - Downloads Python 3.11 (embedded, won't touch your system Python)
   - Installs PyTorch with CUDA 12.1
   - Installs SPAG-4D and its dependencies
   - Clones DAP (Depth Any Panorama) architecture
   - Clones DA360 (Depth Anything 360) architecture
   - Clones ML-SHARP (Apple) for optional refined mode
   - Downloads DAP model weights (~1.5 GB)
   - Downloads DA360 model weights (~1.3 GB)

4. Wait for **"Installation Complete!"** and press any key to close.

This takes 5--15 minutes depending on your internet speed. If it fails partway through (network drop, etc.), just run `install.bat` again -- it picks up where it left off.

---

## Step 2: Run

1. Double-click **`run.bat`**.

2. A terminal appears and your browser opens to **http://localhost:7860**.

3. A demo panorama is pre-loaded. Click **Convert** to test.

4. When processing finishes, the 3D scene appears in the viewer.

**Viewer controls:**
| Action | Control |
|--------|---------|
| Orbit | Left-click drag |
| Pan | Right-click drag |
| Zoom | Scroll wheel |
| Reset | Click the Reset View button |

Keep the terminal window open while using the app. Press `Ctrl+C` in the terminal to stop the server.

---

## Pipeline Modes

SPAG-4D v3.0 supports two pipeline modes, selectable in the web UI or CLI:

### SPAG Mode (Default)

Fast depth-to-Gaussian conversion using spherical projection. Colors come directly from the panorama pixels. Adjustable density via the **Stride** parameter.

- **Stride 1**: One Gaussian per pixel (~1.3M splats) -- maximum quality
- **Stride 2**: Quarter density (~350K splats) -- default, good balance
- **Stride 4**: Fast preview (~85K splats)
- **Stride 8**: Ultra-fast (~21K splats)

### SHARP Refined Mode

Enable the **SHARP Refine** checkbox (or `--sharp-refine` on CLI) for per-face neural Gaussian prediction. Slower (~60s+) but produces higher detail per face.

---

## Depth Models

| Model | Description | When to Use |
|-------|-------------|-------------|
| **DAP** | Metric radial depth designed for 360° images | Default -- reliable, consistent scale |
| **DA360** | Scale-invariant disparity with circular padding | Seamless pole handling, research use |

Select the depth model from the **Depth Model** dropdown in the web UI, or use `--depth-model da360` on the command line.

---

## What Gets Downloaded

| Component | Size | Purpose |
|-----------|------|---------|
| **Python 3.11 Embedded** | ~30 MB | Self-contained Python runtime |
| **PyTorch + CUDA 12.1** | ~2.5 GB | GPU computation framework |
| **DAP model weights** | ~1.5 GB | 360° metric depth estimation |
| **DA360 model weights** | ~1.3 GB | 360° scale-invariant depth estimation |
| **ML-SHARP weights** | ~3 GB | Per-face Gaussian prediction (downloaded on first use of SHARP mode) |

Models are cached locally in `~/.cache/spag4d/`. After the first run, startup is fast.

---

## Converting Your Own Panoramas

Any equirectangular (2:1 aspect ratio) panorama works. Common sources:

- **Insta360** / **Ricoh Theta** / **GoPro Max** -- export as equirectangular JPEG
- **Google Street View** -- download panoramas with third-party tools
- **Polyhaven** / **HDRI Haven** -- free HDRIs (convert to JPEG/PNG first)
- **Blender / Unity** -- render a 360° equirectangular camera

Upload your image through the web UI or use the command line:

```
python_embed\python.exe -m spag4d convert your_panorama.jpg output.ply
python_embed\python.exe -m spag4d convert your_panorama.jpg output.ply --stride 1  # max quality
python_embed\python.exe -m spag4d convert your_panorama.jpg output.ply --depth-model da360  # DA360
python_embed\python.exe -m spag4d convert your_panorama.jpg output.ply --sharp-refine  # SHARP mode
```

---

## Adjusting Quality

| Setting | What to change | Effect |
|---------|---------------|--------|
| **Stride** | Lower value (1 = max) | More Gaussians, higher detail |
| **Stride** | Higher value (4, 8) | Faster, fewer splats |
| **Depth Model** | Switch to DA360 | Different depth characteristics, no seam at 360° boundary |
| **SHARP Refine** | Enable checkbox | Neural per-face refinement, much slower but higher detail |
| **Sky Threshold** | Lower the value | Removes more distant geometry (sky, clouds) |
| **Outlier Pruning** | Increase toward 1.0 | Removes stray floating Gaussians |

---

## Troubleshooting

### "git is not recognized"

Git isn't installed or isn't on your PATH. Install it from [git-scm.com](https://git-scm.com/downloads), then close and reopen your terminal before running `install.bat` again.

### Install fails or hangs

Network interruptions can cause pip to fail silently. Close the terminal and run `install.bat` again. It skips steps that already completed.

### "No module named 'spag4d.dap_arch.DAP.networks'"

The DAP submodule didn't clone properly. Open a terminal in the SPAG-4D folder and run:

```
git submodule update --init --recursive
```

### DA360 model not available

The DA360 architecture wasn't cloned. Run:

```
git clone https://github.com/Insta360-Research-Team/DA360 spag4d\da360_arch\DA360
```

### SHARP not found warning

ML-SHARP didn't install. Run:

```
python_embed\Scripts\pip.exe install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip
```

### CUDA out of memory

- In SPAG mode: increase the **Stride** value (4 or 8)
- In SHARP mode: reduce **Face Size** to 768 or use Cubemap (6 faces)

### Port 7860 already in use

Another application (or a previous SPAG-4D instance) is using the port. Either close it, or edit `run.bat` and change `7860` to another number like `7861`.

### Firewall popup

Windows Firewall may ask to allow network access. Click **Allow** -- the server only listens on your local machine (`127.0.0.1`), not the internet.

---

## Folder Structure

After installation:

```
SPAG-4D/
├── python_embed/              Embedded Python 3.11 (created by install.bat)
├── spag4d/                    Application source code
│   ├── dap_arch/DAP/          DAP depth model architecture
│   └── da360_arch/DA360/      DA360 depth model architecture
├── static/                    Web interface
├── ml-sharp/                  Apple ML-SHARP (cloned by install.bat)
├── TestImage/                 Demo panorama
├── install.bat                Run once to set up
├── run.bat                    Run to start the app
├── api.py                     Web server
└── README.md                  Documentation

~/.cache/spag4d/               Model weight cache
├── model.pth                  DAP weights (~1.5 GB)
└── DA360_large.pth            DA360 weights (~1.3 GB)
```

---

## Updating

1. Download the new release `.zip`.
2. Extract over the existing folder (overwrite files when prompted).
3. Run `install.bat` again to pick up any new dependencies.

Your `python_embed/`, model caches, and `ml-sharp/` folders are preserved.

---

## Uninstalling

Delete the SPAG-4D folder. That's it -- nothing is installed system-wide. Model caches in `~/.cache/huggingface/` and `~/.cache/spag4d/` can also be deleted to reclaim disk space.
