# Design Doc: Integrating UniSHARP Into SPAG4d's SHARP 360 Pipeline

**Project:** SPAG4d
**Target repo:** `cedarconnor/SPAG4d`
**Target integration area:** `spag4d/sharp360.py`, `spag4d/core.py`
**Related upstream:** [Insta360-Research-Team/UniSHARP](https://github.com/Insta360-Research-Team/UniSHARP) (MIT)
**Doc version:** 2.0
**Date:** 2026-06-11

> **v2 note.** This revision supersedes the v1.0 doc. It is grounded against the
> *actual* UniSHARP source (`scripts/infer_unisharp.py`, `unisharp/utils/gaussians.py`)
> rather than assumed behavior. Sections that v1 got materially wrong — license
> status, camera-JSON handling, the `--save-ply` flag, the output PLY format, and
> the `max-long-edge` "knob" — are corrected here and flagged inline with
> **[CORRECTED]**.

---

## 1. Executive Summary

SPAG4d already has a working SHARP 360 path that converts a 2:1 equirectangular
(ERP) panorama into a merged 3D Gaussian Splat by extracting perspective faces,
running SHARP per face, clipping each face into an ownership region, aligning
depths to DA360, rotating into world space, merging, and exporting through
SHARP's PLY writer.

UniSHARP is a natural sibling backend. It extends SHARP-style monocular Gaussian
view synthesis to *universal* cameras (perspective, wide-FoV, fisheye, and
panorama) using UniK3D for ray/feature prediction. Critically for SPAG4d, it can
consume a **native ERP panorama directly** via `--camera panorama`, skipping the
cubemap decomposition entirely.

The integration strategy is unchanged in spirit from v1:

```text
spag4d/sharp360.py
  convert_sharp360(..., backend="sharp" | "unisharp" | "hybrid")
    backend="sharp":      existing per-face SHARP behavior (unchanged)
    backend="unisharp":   ERP -> UniSHARP native panorama inference
                              -> optional DA360 scale alignment -> PLY
    backend="hybrid":     SHARP face pass + UniSHARP native pass
                              -> merge / hole-fill -> PLY   (future)
```

User-facing CLI stays simple:

```bash
python -m spag4d convert pano.jpg out.ply --generator sharp360 --sharp-backend unisharp
```

The recommended first milestone is a **subprocess adapter** that calls UniSHARP's
`scripts/infer_unisharp.py` inside its own conda environment, then copies the
resulting `gaussians.ply` into SPAG4d's output path. UniSHARP must not be imported
into the SPAG4d process — its dependency stack (Python 3.12, torch 2.8, a
from-source UniK3D) is heavy and isolation-sensitive.

---

## 2. Ground Truth: What UniSHARP Actually Does

This section records verified facts from the upstream source. Everything in later
sections depends on these.

### 2.1 License **[CORRECTED]**

UniSHARP ships under the **MIT license**. It is *not* noncommercial/research-only.
The v1 doc's license warnings, "do not vendor," and commercial-safety caveats were
based on a wrong assumption.

The real licensing consideration is the **dependency chain**, not UniSHARP itself:

- **UniK3D** ([lpiccinelli-eth/UniK3D](https://github.com/lpiccinelli-eth/UniK3D)) —
  cloned into `Unisharp/UniK3D`. Required for all inference paths. Verify its
  license before any commercial Sphere use.
- **3DGEER** ([boschresearch/3dgeer](https://github.com/boschresearch/3dgeer)) —
  cloned into `Unisharp/3dgeer`. Required **only for fisheye rendering**.
  **Not needed for the panorama path.** Can be skipped entirely for SPAG4d.
- **gsplat**, **SHARP (apple/ml-sharp)** — acknowledged upstream deps.

### 2.2 Inference Entry Point: `scripts/infer_unisharp.py`

Verified argparse surface:

```text
--checkpoint PATH            (required)  step_XXXXXXX.pt
--image PATH                 single image
--image-list PATH            newline-delimited list
--image-dir PATH             directory of images
--out-dir PATH               default: <repo>/outputs/inference
--device STR                 default: cuda:0 if available else cpu
--max-images INT             default: 0 (0 = all)
--save-ply                   store_true, DEFAULT OFF       <-- critical
--camera-json PATH           calibrated intrinsics (NOT needed for ERP)
--camera-intrinsics f...     explicit pinhole K
--camera-params f...         explicit Fisheye624 params
--camera {auto,perspective,pinhole,fisheye,panorama,erp}   default: auto
--low-pass-filter-eps FLOAT  default: 0.0
```

### 2.3 The `--save-ply` Flag **[CORRECTED]**

`--save-ply` is `action="store_true"` and defaults to **off**. Without it, the
script writes GIFs and `metadata.json` but **no PLY at all**. Any adapter that
forgets this flag will fail the "no gaussians.ply found" check on every run. This
was the single largest latent bug in the v1 design.

**The adapter must always pass `--save-ply`.**

### 2.4 Camera Selection **[CORRECTED]**

The script has a first-class `--camera panorama` override. For a known 2:1 ERP,
pass it directly. The elaborate `write_panorama_camera_json()` helper from v1
(with dual basename/abspath keys) is **unnecessary** and is dropped from this
design.

Camera resolution precedence (from `_should_load_panorama_native`):

1. `--camera panorama|erp` → forced panorama.
2. `--camera perspective|pinhole|fisheye` → forced non-panorama.
3. `--camera-json` entry's `camera` field, if present.
4. Aspect-ratio auto-detection: panorama when aspect ∈ `[1.9, 2.1]`
   (`PANORAMA_ASPECT_MIN/MAX`).

`--camera-json` exists to supply *calibrated intrinsics* (fx/fy/cx/cy, fisheye624
params). A panorama needs none of that, so SPAG4d's panorama backend will not
write a camera JSON.

### 2.5 Resolution Cap Is Hardcoded **[CORRECTED]**

`PANORAMA_MAX_LONG_EDGE = 1536` is a module constant in `infer_unisharp.py`, not a
CLI argument. There is no flag to change it. The v1 `--unisharp-max-long-edge`
option bound to nothing.

**Decision:** drop `--unisharp-max-long-edge` from the public CLI. If a different
cap is ever needed, it requires patching the upstream constant; document that as a
repo-level modification, not a SPAG4d option.

### 2.6 Output Layout **[CORRECTED]**

For each input image the script writes a sample directory:

```text
<out-dir>/<slug>/
  gaussians.ply          (only if --save-ply)
  forward.gif            (always)
  rotate.gif             (always)
  metadata.json          (always)
  forward_erp/           (per-view PNGs)
  rotate_erp/
  rotate_cubemap_faces/<up|back|left|front|right|down>/
```

The **slug is not the image stem**. It is:

```python
slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{image.parent.name}_{image.stem}")
```

So `D:/panos/pano.jpg` → `panos_pano/gaussians.ply`. Locate the PLY by glob
(`**/gaussians.ply`) rather than assuming the directory name.

### 2.7 GIFs Always Render

`forward.gif` and `rotate.gif` (10 views each: forward 0.2 m, rotate 0.1 m radius)
are produced on **every** call with no skip flag in the current commit. This is
non-trivial compute paid per panorama even when only the PLY is wanted. If
throughput matters, a small local patch to gate GIF rendering is worth
considering (see §11.3).

### 2.8 Output PLY Format **[CORRECTED]**

The output is **standard INRIA 3DGS PLY**, confirmed by UniSHARP's own
`load_ply()` reading its own writer. Per-vertex fields:

```text
x, y, z                     position (world space)
f_dc_0, f_dc_1, f_dc_2      SH degree-0 color
scale_0, scale_1, scale_2   log-scale (exp on load)
rot_0, rot_1, rot_2, rot_3  quaternion, wxyz order
opacity                     logit (sigmoid on load)
```

Convention answers to v1's "Open Questions §18":

| Question | Answer (from source) |
|---|---|
| Color encoding | SH deg-0. `rgb = sh0 * 0.28209479 + 0.5` |
| Opacity | logits; `sigmoid()` on load |
| Scale | log; `exp()` on load |
| Quaternion order | **wxyz** (`rot_0`=w) |
| World convention | gaussians pre-transformed to world via `transform_gaussians_to_world` (UniK3D camera-at-origin frame) |

**The one real compatibility risk:** the writer appends extra PLY *supplement
elements* — `extrinsic`, `intrinsic`, `color_space`, `image_size` — alongside the
`vertex` element. Strict third-party loaders may choke on these. The core vertex
fields themselves are vanilla and load fine in SuperSplat / GaussianSplats3D.

**Honor the `color_space` supplement** (`sRGB` vs `linearRGB`) rather than
assuming — UniSHARP's own loader branches on it (`sRGB2linearRGB`).

### 2.9 Checkpoints

Released checkpoints live at HuggingFace
[`Insta360-Research/Unisharp`](https://huggingface.co/Insta360-Research/Unisharp/tree/main)
as `step_XXXXXXX.pt`. The loader accepts a dict checkpoint and an optional
sidecar `config.json` placed next to it. No fixed filename is required.

### 2.10 Capability Envelope (Strategic)

UniSHARP is **monocular, single-image, small-baseline** novel-view synthesis
(forward 0.2 m, rotate 0.1 m radius — verified constants). It will not produce
large-baseline walkable 3DGS from one panorama any more than the existing SHARP
path does. For SPAG4d it is a **native-ERP / quality** upgrade over cubemap-SHARP,
not a capability jump in camera travel range. Plan hybrid work around UniSHARP's
real edge: pole (nadir/zenith) and seam regions where per-face cubemap SHARP is
weakest.

---

## 3. Goals and Non-Goals

### 3.1 Functional Goals

1. Add UniSHARP support without altering existing SHARP 360 behavior.
2. Native ERP inference through UniSHARP via `--camera panorama`.
3. Preserve `--generator sharp360` default behavior (`backend="sharp"`).
4. Add `--sharp-backend {sharp,unisharp,hybrid}`.
5. Run UniSHARP via subprocess in its own environment (no in-process import).
6. Preserve output compatibility with SPAG4d's viewer/export stack.
7. Save useful debug artifacts (raw PLY, metadata, optionally GIFs).
8. Return a stats dict compatible with `convert_sharp360`.

### 3.2 Non-Goals (Initial Milestones)

1. Retraining or modifying UniSHARP internals.
2. Replacing the existing SHARP face pipeline.
3. In-process Python integration of UniSHARP.
4. Full hybrid merging.
5. Auto-installing UniSHARP's dependency stack from SPAG4d.

---

## 4. User-Facing API

### 4.1 CLI Options **[CORRECTED — pruned]**

```bash
--sharp-backend {sharp,unisharp,hybrid}     # default: sharp
--unisharp-repo PATH                         # clone of Insta360 UniSHARP
--unisharp-python PATH                       # python.exe of unisharp conda env
--unisharp-checkpoint PATH                   # step_XXXXXXX.pt
--unisharp-scale-align {none,global,da360_grid}   # default: global
--unisharp-format-mode {copy,convert}        # default: copy
--unisharp-save-debug                        # keep raw dir, gifs, metadata
--unisharp-raw-output-dir PATH               # persist UniSHARP working dir
```

Removed from v1: `--unisharp-camera-json` (panorama needs none) and
`--unisharp-max-long-edge` (binds to nothing; constant is hardcoded upstream).

Recommended defaults:

```python
sharp_backend       = "sharp"
unisharp_repo       = None        # or env SPAG4D_UNISHARP_REPO
unisharp_python     = None        # or env SPAG4D_UNISHARP_PYTHON
unisharp_checkpoint = None        # or env SPAG4D_UNISHARP_CHECKPOINT
unisharp_scale_align = "global"
unisharp_format_mode = "copy"
unisharp_save_debug  = False
```

Optional environment fallbacks:

```bash
SPAG4D_UNISHARP_REPO=/path/to/UniSHARP
SPAG4D_UNISHARP_PYTHON=/path/to/unisharp/python
SPAG4D_UNISHARP_CHECKPOINT=/path/to/step_0100000.pt
```

### 4.2 Example Commands

Existing behavior (unchanged):

```bash
python -m spag4d convert pano.jpg out.ply --generator sharp360
```

Native UniSHARP panorama path:

```bash
python -m spag4d convert pano.jpg out_unisharp.ply \
  --generator sharp360 \
  --sharp-backend unisharp \
  --unisharp-repo D:/repos/UniSHARP \
  --unisharp-python D:/envs/unisharp/python.exe \
  --unisharp-checkpoint D:/models/unisharp/step_0100000.pt
```

Optional dedicated alias (maps internally to `sharp360` + `backend="unisharp"`):

```bash
python -m spag4d convert pano.jpg out.ply --generator unisharp360
```

Hybrid (future):

```bash
python -m spag4d convert pano.jpg out_hybrid.ply \
  --generator sharp360 --sharp-backend hybrid \
  --unisharp-repo D:/repos/UniSHARP \
  --unisharp-python D:/envs/unisharp/python.exe \
  --unisharp-checkpoint D:/models/unisharp/step_0100000.pt \
  --unisharp-scale-align da360_grid
```

---

## 5. Code Organization

### 5.1 New Files **[CORRECTED — slimmer]**

```text
spag4d/
  unisharp360.py        # high-level SPAG4d-facing wrapper
  unisharp_adapter.py   # subprocess runner around infer_unisharp.py
  unisharp_format.py    # PLY inspect / count / copy / convert
```

`unisharp_camera.py` from v1 is **deleted** — the panorama path uses
`--camera panorama` and needs no camera JSON.

Optional later:

```text
spag4d/
  hybrid_merge.py
  gaussian_compare.py
```

### 5.2 Responsibilities

**`unisharp360.py`** — validate input, prepare work dir, call adapter, locate raw
PLY, copy or convert format, optionally scale-align, return stats dict.

**`unisharp_adapter.py`** — build and run the subprocess (with correct `cwd`),
capture stdout/stderr, raise readable errors, return result paths.

**`unisharp_format.py`** — inspect PLY fields, count vertices, copy raw PLY to
output (handling the supplement-element risk), and later convert into a
SPAG4d-native PLY honoring `color_space`.

---

## 6. `unisharp_adapter.py` — Detailed Design

This is the only non-trivial new file for Milestone 1.

```python
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def run_unisharp_inference(
    image_path: str,
    out_dir: str,
    repo_dir: str,
    checkpoint_path: str,
    python_exe: Optional[str] = None,
    camera: str = "panorama",
    device: str = "cuda:0",
    save_ply: bool = True,
    max_images: int = 1,
    extra_args: Optional[list[str]] = None,
    timeout_s: Optional[float] = None,
) -> dict:
    """Invoke UniSHARP's infer_unisharp.py as a subprocess.

    Returns dict with returncode, stdout, stderr, out_dir, ply_path, metadata_path.
    """
    repo = Path(repo_dir)
    script = repo / "scripts" / "infer_unisharp.py"
    if not script.exists():
        raise FileNotFoundError(f"UniSHARP inference script not found: {script}")

    python = python_exe or sys.executable

    cmd = [
        python, str(script),
        "--checkpoint", str(checkpoint_path),
        "--image", str(image_path),
        "--out-dir", str(out_dir),
        "--camera", camera,
        "--device", device,
        "--max-images", str(max_images),
    ]
    if save_ply:
        cmd.append("--save-ply")          # REQUIRED or no PLY is written
    if extra_args:
        cmd.extend(extra_args)

    LOGGER.info("Running UniSHARP: %s", " ".join(cmd))

    # cwd MUST be the repo root: the script does sys.path.insert(REPO_ROOT),
    # writes checkpoints/torchhub there, and expects ./UniK3D as a sibling.
    proc = subprocess.run(
        cmd, cwd=str(repo),
        capture_output=True, text=True, timeout=timeout_s,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "UniSHARP inference failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )

    out = Path(out_dir)
    ply_candidates = sorted(out.glob("**/gaussians.ply"))
    meta_candidates = sorted(out.glob("**/metadata.json"))
    if not ply_candidates:
        raise FileNotFoundError(
            f"UniSHARP completed but no gaussians.ply under {out}. "
            "Confirm --save-ply was passed and inference produced a sample dir."
        )
    if len(ply_candidates) > 1:
        LOGGER.warning("Multiple PLYs found; using first: %s", ply_candidates[0])

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_dir": str(out),
        "ply_path": str(ply_candidates[0]),
        "metadata_path": str(meta_candidates[0]) if meta_candidates else None,
    }
```

### 6.1 Why a Separate Environment Is Mandatory

UniSHARP targets Python 3.12, torch 2.8 / torchvision 0.23, and a from-source
UniK3D (which builds its own CUDA ops). Importing this into SPAG4d's process risks
torch/CUDA ABI conflicts. The subprocess-into-its-own-conda-env approach via
`--unisharp-python` is the correct isolation boundary. Set `cwd=repo_dir` (or
`env["PYTHONPATH"]=repo_dir`) so the script's `sys.path.insert(REPO_ROOT)` and its
relative `UniK3D/`, `checkpoints/torchhub/` paths resolve.

---

## 7. `unisharp360.py` — Detailed Design

```python
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)


def convert_unisharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    unisharp_repo: Optional[str],
    unisharp_python: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    scale_align: str = "global",
    format_mode: str = "copy",
    save_debug: bool = False,
    raw_output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    t0 = time.time()

    # ---- resolve env fallbacks -------------------------------------------
    unisharp_repo = unisharp_repo or os.environ.get("SPAG4D_UNISHARP_REPO")
    unisharp_python = unisharp_python or os.environ.get("SPAG4D_UNISHARP_PYTHON")
    checkpoint_path = checkpoint_path or os.environ.get("SPAG4D_UNISHARP_CHECKPOINT")

    # ---- validation -------------------------------------------------------
    if not unisharp_repo:
        raise ValueError(
            "UniSHARP backend requires --unisharp-repo (or SPAG4D_UNISHARP_REPO) "
            "pointing to a local clone of Insta360-Research-Team/UniSHARP."
        )
    repo = Path(unisharp_repo)
    if not repo.exists():
        raise FileNotFoundError(f"UniSHARP repo not found: {repo}")
    if not checkpoint_path:
        raise ValueError(
            "UniSHARP backend requires --unisharp-checkpoint "
            "(or SPAG4D_UNISHARP_CHECKPOINT)."
        )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"UniSHARP checkpoint not found: {checkpoint_path}")

    with Image.open(input_path) as im:
        w, h = im.size
    if abs((w / h) - 2.0) > 0.05:
        raise ValueError(
            f"UniSHARP panorama mode expects a 2:1 ERP image. Got {w}x{h}. "
            "Use --sharp-backend sharp for non-ERP inputs."
        )

    # ---- working directory ------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_output_dir:
        work_dir = Path(raw_output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="spag4d_unisharp_")
        work_dir = Path(tmp_ctx.name)

    try:
        if progress_callback:
            progress_callback("unisharp_inference", 0, 1)

        from .unisharp_adapter import run_unisharp_inference
        run = run_unisharp_inference(
            image_path=input_path,
            out_dir=str(work_dir / "unisharp_out"),
            repo_dir=str(repo),
            checkpoint_path=str(checkpoint_path),
            python_exe=unisharp_python,
            camera="panorama",
            device=("cuda:0" if device.type == "cuda" else "cpu"),
            save_ply=True,
            max_images=1,
        )
        raw_ply = Path(run["ply_path"])

        if progress_callback:
            progress_callback("unisharp_inference", 1, 1)

        # ---- format handling ---------------------------------------------
        from .unisharp_format import (
            copy_unisharp_ply_to_output,
            convert_unisharp_ply_to_spag,
        )
        if format_mode == "copy":
            stats = copy_unisharp_ply_to_output(str(raw_ply), str(out_path))
        elif format_mode == "convert":
            stats = convert_unisharp_ply_to_spag(str(raw_ply), str(out_path))
        else:
            raise ValueError(f"Unknown format_mode {format_mode!r}")

        # ---- optional scale alignment ------------------------------------
        # scale_align in {"none","global","da360_grid"} — see §9. Applied
        # in-place to the written PLY (or pre-write in convert mode).
        # Milestone 1 ships scale_align == "none"/"global" no-op-safe.

        # ---- debug artifacts ---------------------------------------------
        if save_debug:
            dbg = out_path.parent / (out_path.stem + "_unisharp_debug")
            dbg.mkdir(parents=True, exist_ok=True)
            for art in ("forward.gif", "rotate.gif", "metadata.json"):
                src = raw_ply.parent / art
                if src.exists():
                    shutil.copy2(src, dbg / art)
            shutil.copy2(raw_ply, dbg / "raw_gaussians.ply")

        return {
            "num_gaussians": stats["num_gaussians"],
            "num_faces": 0,                       # native ERP: no faces
            "output_path": str(out_path),
            "processing_time": time.time() - t0,
            "backend": "unisharp",
        }
    finally:
        if tmp_ctx is not None and not save_debug:
            tmp_ctx.cleanup()
```

Stats dict keys (`num_gaussians`, `num_faces`, `output_path`) match the existing
`convert_sharp360` contract so `core.py` needs no special-casing.

---

## 8. `unisharp_format.py` — Detailed Design

```python
from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)

CORE_VERTEX_FIELDS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
    + ["opacity"]
)


def inspect_ply_fields(ply_path: str) -> dict:
    """Report element names, vertex field names, and supplement elements."""
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    elements = {el.name: [p.name for p in el.properties] for el in ply.elements}
    vertex_fields = elements.get("vertex", [])
    supplements = [name for name in elements if name != "vertex"]
    return {
        "elements": elements,
        "vertex_fields": vertex_fields,
        "supplement_elements": supplements,   # extrinsic/intrinsic/color_space/image_size
        "has_core_fields": all(f in vertex_fields for f in CORE_VERTEX_FIELDS),
        "color_space": _read_color_space(ply),
    }


def _read_color_space(ply) -> str | None:
    for el in ply.elements:
        if el.name == "color_space" or "color_space" in [p.name for p in el.properties]:
            try:
                return str(el["color_space"][0])
            except Exception:
                return None
    return None


def count_ply_vertices(ply_path: str) -> int:
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    for el in ply.elements:
        if el.name == "vertex":
            return int(el.count)
    return 0


def copy_unisharp_ply_to_output(src_path: str, dst_path: str) -> dict:
    """Milestone-1 path: copy raw PLY verbatim, just count vertices.

    Keeps UniSHARP's supplement elements intact. Works in viewers that
    tolerate extra PLY elements (SuperSplat, GaussianSplats3D).
    """
    shutil.copy2(src_path, dst_path)
    n = count_ply_vertices(dst_path)
    info = inspect_ply_fields(dst_path)
    if info["supplement_elements"]:
        LOGGER.info(
            "UniSHARP PLY carries supplement elements %s; "
            "strict loaders may need format_mode='convert'.",
            info["supplement_elements"],
        )
    return {"num_gaussians": n, "ply_info": info}


def convert_unisharp_ply_to_spag(src_path: str, dst_path: str) -> dict:
    """Milestone-3 path: rewrite as a clean INRIA 3DGS PLY.

    Steps:
      1. Read vertex element only; drop supplement elements.
      2. If color_space supplement == 'linearRGB' and SPAG4d expects sRGB
         (or vice-versa), convert f_dc accordingly. Default: preserve.
      3. Re-emit vertex with exactly CORE_VERTEX_FIELDS in SPAG4d's
         field order. Scales stay log, opacity stays logit, quats stay wxyz.
      4. Apply any world-axis convention fix vs SPAG4d's SHARP frame
         (verify empirically before enabling).
    """
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(src_path)
    vtx = next(el for el in ply.elements if el.name == "vertex")
    data = {name: np.asarray(vtx[name]) for name in CORE_VERTEX_FIELDS if name in vtx}

    missing = [f for f in CORE_VERTEX_FIELDS if f not in data]
    if missing:
        raise KeyError(f"UniSHARP PLY missing core fields: {missing}")

    dtype = [(name, "f4") for name in CORE_VERTEX_FIELDS]
    arr = np.empty(len(data["x"]), dtype=dtype)
    for name in CORE_VERTEX_FIELDS:
        arr[name] = data[name].astype("f4")

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(dst_path)
    return {"num_gaussians": int(len(arr)), "converted": True}
```

---

## 9. Scale Alignment (Milestones 4-5)

UniSHARP output is metric-ish (UniK3D-derived) but will not match SPAG4d's
SHARP/DA360 world scale. Three modes:

- **`none`** — emit UniSHARP scale unchanged. Useful for isolated inspection.
- **`global`** (default) — single scalar `s` matching median UniSHARP radius to a
  reference median depth (DA360 on the same ERP). Apply `s` to positions and add
  `log(s)` to `scale_*` (log-scale convention). Cheap, robust, no distortion.
- **`da360_grid`** — project UniSHARP centers back to ERP, sample DA360 disparity,
  estimate a smooth local scale field, apply per-gaussian. Higher fidelity, real
  research effort; defer until `global` is validated.

Global alignment sketch:

```python
import numpy as np

def global_scale_factor(unisharp_xyz: np.ndarray, ref_depth_median_m: float) -> float:
    r = np.linalg.norm(unisharp_xyz, axis=1)
    r_med = float(np.median(r[r > 0]))
    if r_med <= 0:
        return 1.0
    return float(ref_depth_median_m) / r_med
```

---

## 10. Changes to Existing Files

### 10.1 `spag4d/sharp360.py`

Extend the signature and dispatch by backend **after** panorama load/validation:

```python
def convert_sharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    side_count: int = 6,
    overlap_degrees: float = 10.0,
    seedvr2_upscale: bool = False,
    seedvr2_config: Optional["SeedVR2Config"] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    include_caps: bool = True,
    cap_fov_degrees: float = 125.0,
    seam_latitude_degrees: float = 30.0,
    # --- new ---
    backend: str = "sharp",
    unisharp_repo: Optional[str] = None,
    unisharp_python: Optional[str] = None,
    unisharp_checkpoint: Optional[str] = None,
    unisharp_scale_align: str = "global",
    unisharp_format_mode: str = "copy",
    unisharp_save_debug: bool = False,
    unisharp_raw_output_dir: Optional[str] = None,
) -> dict:
    backend = str(backend).lower().strip()
    if backend not in {"sharp", "unisharp", "hybrid"}:
        raise ValueError(
            f"Unknown sharp360 backend {backend!r}. "
            "Expected 'sharp', 'unisharp', or 'hybrid'."
        )

    if backend == "unisharp":
        from .unisharp360 import convert_unisharp360
        return convert_unisharp360(
            input_path=input_path,
            output_path=output_path,
            device=device,
            unisharp_repo=unisharp_repo,
            unisharp_python=unisharp_python,
            checkpoint_path=unisharp_checkpoint,
            scale_align=unisharp_scale_align,
            format_mode=unisharp_format_mode,
            save_debug=unisharp_save_debug,
            raw_output_dir=unisharp_raw_output_dir,
            progress_callback=progress_callback,
        )

    if backend == "hybrid":
        raise NotImplementedError(
            "sharp360 backend='hybrid' is planned but not implemented. "
            "Use 'sharp' or 'unisharp'."
        )

    # backend == "sharp": existing per-face pipeline, untouched.
    ...
```

### 10.2 `spag4d/core.py`

Route both generator names and inject the new args:

```python
if active_generator in ("sharp360", "unisharp360"):
    from .sharp360 import convert_sharp360
    from .seedvr2 import SeedVR2Config

    effective_backend = sharp_backend
    if active_generator == "unisharp360":
        effective_backend = "unisharp"

    seedvr2_cfg = SeedVR2Config() if seedvr2_upscale else None

    result_dict = convert_sharp360(
        input_path=str(input_path),
        output_path=str(output_path),
        device=self.device,
        side_count=side_count,
        seedvr2_upscale=seedvr2_upscale,
        seedvr2_config=seedvr2_cfg,
        include_caps=sharp_include_caps,
        cap_fov_degrees=sharp_cap_fov,
        seam_latitude_degrees=sharp_seam_latitude,
        backend=effective_backend,
        unisharp_repo=unisharp_repo,
        unisharp_python=unisharp_python,
        unisharp_checkpoint=unisharp_checkpoint,
        unisharp_scale_align=unisharp_scale_align,
        unisharp_format_mode=unisharp_format_mode,
        unisharp_save_debug=unisharp_save_debug,
        unisharp_raw_output_dir=(
            str(unisharp_raw_output_dir) if unisharp_raw_output_dir else None
        ),
    )

    file_size = Path(output_path).stat().st_size
    return ConversionResult(
        output_path=str(output_path),
        splat_count=result_dict["num_gaussians"],
        file_size=file_size,
        processing_time=result_dict.get("processing_time", 0.0),
        depth_range=result_dict.get("depth_range", (0.0, 0.0)),
        panorama_size=(W, H),
    )
```

Add the corresponding `convert()` parameters (`sharp_backend`, `unisharp_*`) with
the defaults from §4.1.

---

## 11. Operational Notes

### 11.1 First-Run Setup (one-time, manual)

```bash
git clone https://github.com/Insta360-Research-Team/UniSHARP.git
cd Unisharp
conda create -n unisharp python=3.12 -y
conda activate unisharp
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
git clone https://github.com/lpiccinelli-eth/UniK3D.git UniK3D
# 3dgeer only needed for fisheye; skip for panorama
# download a checkpoint from HuggingFace Insta360-Research/Unisharp
```

### 11.2 A6000 / Ampere Notes

The 1536 long-edge ERP cap keeps memory modest; the A6000's 48 GB is comfortable.
torch 2.8 cu12x wheels run on Ampere without issue. Keep this env fully separate
from SPAG4d's torch to avoid CUDA ABI clashes.

### 11.3 Optional GIF Suppression

`forward.gif` + `rotate.gif` render on every call (20 novel views total). If batch
throughput matters, a one-line guard around `_save_gif(...)` calls in
`infer_unisharp.py` (gated by a new `--no-gifs` flag) removes that cost. Treat as a
local fork patch, documented separately, not a SPAG4d feature.

---

## 12. Error Handling

| Condition | Message |
|---|---|
| Missing repo | "UniSHARP backend requires --unisharp-repo pointing to a local clone of Insta360-Research-Team/UniSHARP." |
| Missing checkpoint | "UniSHARP backend requires --unisharp-checkpoint (a step_XXXXXXX.pt from HuggingFace Insta360-Research/Unisharp)." |
| Wrong aspect | "UniSHARP panorama mode expects a 2:1 ERP image. Got {w}x{h}. Use --sharp-backend sharp for non-ERP inputs." |
| Subprocess failure | Full cmd + stdout + stderr (see adapter). |
| No PLY found | "UniSHARP completed but no gaussians.ply under {dir}. Confirm --save-ply was passed." |

---

## 13. Testing Plan

### 13.1 Unit Tests

```text
- adapter command construction includes --save-ply and --camera panorama
- adapter sets cwd to repo root
- PLY field inspection detects core fields + supplement elements
- vertex count
- missing repo / missing checkpoint / wrong aspect raise correctly
- convert mode drops supplement elements and preserves core fields
```

### 13.2 Smoke Test

```bash
python -m spag4d convert test_pano.jpg test_unisharp.ply \
  --generator sharp360 --sharp-backend unisharp \
  --unisharp-repo /path/to/UniSHARP \
  --unisharp-python /path/to/unisharp/python \
  --unisharp-checkpoint /path/to/step_0100000.pt \
  --unisharp-save-debug
```

Expect: command completes, output PLY exists with > 0 vertices, debug folder has
`raw_gaussians.ply` + `metadata.json` (+ gifs).

### 13.3 Visual / Comparison Tests

Open output in SuperSplat and SPAG4d's viewer; check orientation, scale, color,
opacity. Then run the same pano through `da360`, `dap`, `pager`,
`sharp360 --sharp-backend sharp`, and `sharp360 --sharp-backend unisharp` and
compare gaussian count, load success, seam behavior, pole behavior, and
close-object disocclusion.

---

## 14. Milestones **[CORRECTED — reordered]**

1. **M1 — Raw UniSHARP backend.** CLI args, `sharp360` dispatch, `core.py` routing,
   adapter (`--save-ply`, `cwd=repo`), copy mode. *Done when
   `--sharp-backend unisharp` produces a PLY.*
2. **M1.5 — Viewer reality check (NEW).** Open the raw PLY in SuperSplat + SPAG4d
   viewer *before* writing any converter. If the supplement elements load fine and
   orientation is acceptable, convert mode may be deferrable indefinitely. This
   kills-or-confirms M3 for near-zero cost.
3. **M3 — PLY conversion** (promoted ahead of bare inspection). Strip supplement
   elements, honor `color_space`, re-emit clean INRIA PLY. *Done when
   `--unisharp-format-mode convert` opens in strict loaders.*
4. **M4 — Global scale alignment.** Median-radius vs DA360 median depth. *Done when
   UniSHARP scale roughly matches DA360/DAP output.*
5. **M5 — DA360 grid alignment.** Local scale field. *Done when local depth
   consistency improves without visible distortion.*
6. **M6 — Hybrid.** Run both backends; use UniSHARP as nadir/zenith/pole and seam
   filler for the cubemap-SHARP result. *Done when hybrid has fewer pole gaps than
   SHARP and better native-ERP detail than raw UniSHARP.*

---

## 15. Recommended First Patch

```text
spag4d/core.py
  - add sharp_backend + unisharp_* convert() args
  - route active_generator in ("sharp360","unisharp360") with backend mapping

spag4d/sharp360.py
  - add backend + unisharp_* args to convert_sharp360
  - dispatch backend=="unisharp" -> convert_unisharp360
  - NotImplementedError for "hybrid"

spag4d/unisharp360.py      # high-level wrapper (§7)
spag4d/unisharp_adapter.py # subprocess runner (§6) -- MUST pass --save-ply, cwd=repo
spag4d/unisharp_format.py  # copy/count/inspect (§8)
```

Minimum path: implement `run_unisharp_inference` → `copy_unisharp_ply_to_output`
→ `convert_unisharp360` → dispatch in `convert_sharp360` → CLI args → smoke-test
one pano.

---

## 16. Strategic Caveat

UniSHARP is a **native-ERP quality upgrade**, not a camera-travel capability jump.
Its novel-view baseline is small (0.2 m forward / 0.1 m rotate radius). If the
Sphere/SPAG-4D objective is wide walkable movement from a single panorama,
UniSHARP alone will not deliver that — and neither does the existing SHARP path.
The highest-value role for UniSHARP in SPAG4d is (a) cleaner native-ERP geometry
without cubemap seams, and (b) a hybrid pole/seam filler for the existing pipeline.
Scope hybrid work around that strength rather than expecting a larger reconstruction
envelope.
