# SPAG-4D Installation Guide

Step-by-step instructions for the Windows portable install. For Linux/Mac or pip-based setup, see the [README](README.md#manual-setup-linux--mac--developer).

---

## What You Need

- **Windows 10 or 11** (64-bit)
- **NVIDIA GPU** with 8 GB+ VRAM (6 GB minimum at reduced quality)
- **NVIDIA Driver** 525 or newer
- **Git** -- download from [git-scm.com](https://git-scm.com/downloads) and make sure it's on your PATH
- **Internet connection** for the first-time download (~6 GB total)

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
   - Clones ML-SHARP (Apple) and DAP (Depth Any Panorama)

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

## What Gets Downloaded

Two AI models are downloaded automatically on first use:

| Model | Size | What It Does |
|-------|------|--------------|
| **DAP** (Depth Any Panorama) | ~1.5 GB | Estimates metric depth for the full 360° image |
| **ML-SHARP** (Apple) | ~3 GB | Predicts 3D Gaussians (positions, colors, opacities, scales, rotations) from each projected face |

Models are cached locally. After the first run, startup is fast.

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
```

---

## Adjusting Quality

The defaults are tuned for a good balance of quality and speed. If you want to tweak:

| Setting | What to change | Effect |
|---------|---------------|--------|
| **Face Size** | Increase to 1920, 2304, or 3072 | Sharper detail, uses more VRAM |
| **Face Size** | Decrease to 768 | Faster, less VRAM, lower detail |
| **Projection** | Switch to Icosahedral | 20 faces instead of 6 -- better poles, slower |
| **Sky Threshold** | Lower the value | Removes more distant geometry (sky, clouds) |
| **Outlier Pruning** | Increase toward 1.0 | Removes stray floating Gaussians |

### VRAM Guide

| Face Size | VRAM Needed | Quality |
|-----------|-------------|---------|
| 768 | ~3 GB | Workable -- good for testing |
| 1536 | ~6 GB | **Default** -- recommended |
| 1920 | ~8 GB | High |
| 2304 | ~12 GB | Very high |
| 3072 | ~16 GB+ | Maximum detail |

If you get CUDA out-of-memory errors, reduce the face size first.

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

### SHARP not found warning

ML-SHARP didn't install. Run:

```
python_embed\Scripts\pip.exe install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip
```

### CUDA out of memory

Reduce the face size in the web UI settings panel, or from the command line:

```
python_embed\python.exe -m spag4d convert input.jpg output.ply --sharp-cubemap-size 768
```

### Port 7860 already in use

Another application (or a previous SPAG-4D instance) is using the port. Either close it, or edit `run.bat` and change `7860` to another number like `7861`.

### Firewall popup

Windows Firewall may ask to allow network access. Click **Allow** -- the server only listens on your local machine (`127.0.0.1`), not the internet.

### Antivirus blocks downloads

Some antivirus software flags large downloads or new executables. Add the SPAG-4D folder to your antivirus exclusion list.

---

## Folder Structure

After installation:

```
SPAG-4D/
├── python_embed/     Embedded Python 3.11 (created by install.bat)
├── spag4d/           Application source code
├── static/           Web interface
├── ml-sharp/         Apple ML-SHARP (cloned by install.bat)
├── checkpoints/      Cached model weights
├── TestImage/        Demo panorama
├── install.bat       Run once to set up
├── run.bat           Run to start the app
├── api.py            Web server
└── README.md         Documentation
```

---

## Updating

1. Download the new release `.zip`.
2. Extract over the existing folder (overwrite files when prompted).
3. Run `install.bat` again to pick up any new dependencies.

Your `python_embed/`, `checkpoints/`, and `ml-sharp/` folders are preserved.

---

## Uninstalling

Delete the SPAG-4D folder. That's it -- nothing is installed system-wide. Model caches in `~/.cache/huggingface/` and `~/.cache/spag4d/` can also be deleted to reclaim disk space.
