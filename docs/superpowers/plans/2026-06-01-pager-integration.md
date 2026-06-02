# PaGeR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PaGeR as a fourth depth/geometry backend in `spag4d`, peer to DA360/DAP/SHARP360, with optional learned sky-mask and world-frame surface-normal byproducts wired into `scene_analysis` and `scene_filter`.

**Architecture:** PaGeR's unified DA3 cubemap model (`prs-eth/PaGeR`) is vendored in-process under `spag4d/pager_arch/`. A new `PaGeRModel` matches the existing depth-backend contract — `load()` + `predict(image_tensor) → (depth, sky_mask)` — and stashes normals/native-resolution on the instance. `core.py` gains a `pager` dispatch branch that reuses the entire existing depth→Gaussian→PLY flow, passing the two optional channels into `compute_scene_defaults` (sky) and `prune_grazing_angle` (normals). Both plumbing changes are strictly additive: `None` reproduces today's behavior byte-for-byte.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, click CLI, pytest. New deps (`requirements-pager.txt`): `open_clip_torch`, `pytorch360convert`, `omegaconf`, `einops`, `addict`, `trimesh`. Weights via `huggingface_hub` (CC BY-NC 4.0).

**Reference spec:** `docs/superpowers/specs/2026-06-01-pager-integration-design.md`

**Conventions to follow:**
- Run Python with the venv interpreter: `.venv/Scripts/python.exe` (PaGeR/torch only resolve there).
- Mirror `da360_arch/DA360/` + `da360_model.py` for vendoring and the model wrapper.
- Depth backend contract (from `core.py:166-167`): `depth_engine.predict(image_tensor)` where `image_tensor = torch.from_numpy(np.array(img)).to(device)` is `(H,W,3)` uint8 on device, returning `(depth_tensor, mask_or_None)`.
- Converter world ray frame (from `scene_filter.py:422`): `rhat = [sin(phi)cos(theta), cos(phi), -sin(phi)sin(theta)]`, i.e. `view_ray = means / ||means||`.

---

## Phase 1 — Smoke gate + depth-only backend

### Task 1: Vendor PaGeR and declare optional dependencies

**Files:**
- Create: `spag4d/pager_arch/PaGeR/` (vendored upstream, pinned commit)
- Create: `spag4d/pager_arch/__init__.py`
- Create: `requirements-pager.txt`

- [ ] **Step 1: Clone and vendor a pinned commit**

```bash
# from a scratch dir, not the repo
git clone https://github.com/prs-eth/PaGeR.git /tmp/PaGeR
cd /tmp/PaGeR && git rev-parse HEAD   # record this commit hash for the README provenance
# copy source only — NO weights, NO gradio app extras
```

Copy `/tmp/PaGeR/src` and `/tmp/PaGeR/configs` into `D:\SPAG-4D\spag4d\pager_arch\PaGeR\`. Do **not** copy `app.py`, `.git`, or any `*.safetensors`. Record the commit hash in a top-of-file comment.

- [ ] **Step 2: Write the arch wrapper**

```python
# spag4d/pager_arch/__init__.py
"""Vendored prs-eth/PaGeR (DA3 cubemap multi-view). Apache-2.0 code.
Pinned commit: <HASH from Task 1 Step 1>.
Weights (prs-eth/PaGeR) are CC BY-NC 4.0 — non-commercial only.
"""
from pathlib import Path
import sys

_ARCH_ROOT = Path(__file__).parent / "PaGeR"


def _ensure_on_path():
    """PaGeR's modules import as top-level `src.*`; expose its root on sys.path."""
    root = str(_ARCH_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_pager(repo_id: str, cfg, device):
    """Construct the upstream Pager inference wrapper.
    Mirrors da360_arch.build_da360_model in role."""
    _ensure_on_path()
    from src.pager import Pager  # noqa: E402  (vendored module)
    return Pager(repo_id, cfg=cfg, device=device)
```

- [ ] **Step 3: Write the optional requirements file**

```text
# requirements-pager.txt — optional PaGeR depth backend (CC BY-NC 4.0 weights)
# Install into the existing .venv: pip install -r requirements-pager.txt
# numpy<2 already satisfied by core install (1.26.x); do NOT re-pin here.
open_clip_torch
pytorch360convert
omegaconf
einops
addict
trimesh
```

- [ ] **Step 4: Install the deps**

Run: `.venv/Scripts/python.exe -m pip install -r requirements-pager.txt`
Expected: all install with no numpy upgrade (numpy stays 1.26.4). If pip tries to pull numpy>=2, stop and pin `numpy<2` explicitly in the command.

- [ ] **Step 5: Verify the vendored package imports**

Run: `.venv/Scripts/python.exe -c "from spag4d.pager_arch import build_pager; print('ok')"`
Expected: prints `ok` with no ImportError. (This imports the wrapper, not weights.)

- [ ] **Step 6: Commit**

```bash
git add spag4d/pager_arch requirements-pager.txt
git commit -m "feat(pager): vendor prs-eth/PaGeR arch + optional requirements"
```

---

### Task 2: Native-Windows smoke gate (HARD GATE — do not proceed if this fails)

**Files:**
- Create: `scripts/pager_smoke.py` (throwaway verification script, kept for reproducibility)

This resolves the three residual unknowns: native-Windows execution, exact `erp_to_cubemap` signature, and radial-depth confirmation. **No wrapper/dispatch code in later tasks may be trusted until this passes.**

- [ ] **Step 1: Write the smoke script**

```python
# scripts/pager_smoke.py
"""Phase 1 smoke gate: confirm vendored PaGeR runs on native Windows,
pin the erp_to_cubemap signature, and confirm radial ERP depth output.
Run: .venv/Scripts/python.exe scripts/pager_smoke.py <pano.jpg>
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from spag4d.pager_arch import build_pager, _ensure_on_path

_ensure_on_path()
from src.utils.geometry_utils import erp_to_cubemap  # noqa: E402
# NOTE: read src/utils/geometry_utils.py and confirm the REAL signature of
# erp_to_cubemap (face_w / fov / face order). Adjust the call below to match.


def main(pano_path: str):
    device = torch.device("cuda")
    cfg = OmegaConf.load(hf_hub_download("prs-eth/PaGeR", "config.yaml"))
    pager = build_pager("prs-eth/PaGeR", cfg=cfg, device=device)
    pager.get_intrinsics_extrinsics(image_size=cfg.face_size,
                                    fov=getattr(cfg, "cube_fov", 90.0))
    pager.model.to(device).eval()

    img = np.array(Image.open(pano_path).convert("RGB"))      # (H,W,3) uint8
    H, W = img.shape[:2]
    chw = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    chw = chw * 2.0 - 1.0                                      # to [-1,1] per model card
    cubemap = erp_to_cubemap(chw, face_w=cfg.face_size, fov=90.0).unsqueeze(0).to(device)

    pred = pager(cubemap, dtype=torch.float16, skip_heads={"scale_indoor"})
    depth_metric, _ = pager.process_depth_output(pred["depth"][0], pred["sky"][0],
                                                 (H, W), log_scale=pred.get("scale"))
    d = depth_metric.detach().float().cpu().numpy()
    print(f"native depth shape: {d.shape}  (expect ~1024x2048)")
    print(f"depth min/median/max: {np.nanmin(d):.2f} / {np.nanmedian(d):.2f} / {np.nanmax(d):.2f}")
    print(f"pred keys: {list(pred.keys())}")
    print("RADIAL CHECK: pick a feature at known distance and confirm the value matches.")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Read the real `erp_to_cubemap` signature first**

Run: `.venv/Scripts/python.exe -c "import inspect, sys; from spag4d.pager_arch import _ensure_on_path; _ensure_on_path(); from src.utils import geometry_utils as g; print(inspect.signature(g.erp_to_cubemap))"`
Expected: prints the real signature. Edit the `erp_to_cubemap(...)` call in the script to match (arg names, face order `[F,R,B,L,U,D]`).

- [ ] **Step 3: Run the smoke gate on 3 panoramas**

Run (one outdoor exterior, one interior, one natural landscape — use files under `tests/data/` or your capture set):
`.venv/Scripts/python.exe scripts/pager_smoke.py <pano.jpg>`
Expected: completes with no CUDA/import error on native Windows; depth shape ~`(1024, 2048)`; plausible metric depth; `pred keys` include `depth`, `normals`, `sky`, `scale`.

**GATE:** If it crashes on native Windows (open_clip / pytorch360convert / DINOv2), STOP. Document the failure and switch to the WSL2 subprocess fallback (spec §3.2, §8) before any further task. Do not write the wrapper against a runtime that doesn't work.

- [ ] **Step 4: Confirm radial convention**

Eyeball one feature at a roughly known distance (a wall, a doorway) and confirm the printed depth is the straight-line distance to it (radial), not a vertical/planar Z. Note the finding inline in the script's docstring.

- [ ] **Step 5: Commit**

```bash
git add scripts/pager_smoke.py
git commit -m "test(pager): native-Windows smoke gate + erp_to_cubemap/radial confirmation"
```

---

### Task 3: `PaGeRModel` wrapper (depth-only, sky/normals stashed but unused)

**Files:**
- Create: `spag4d/pager_model.py`
- Test: `tests/test_pager_model.py`

- [ ] **Step 1: Write the failing structural test (no weights required)**

```python
# tests/test_pager_model.py
"""PaGeRModel matches the depth-backend contract. Structural tests use a
monkeypatched upstream so they run without the 5.6GB checkpoint or a GPU."""
import numpy as np
import pytest
import torch


def test_predict_returns_depth_and_sky_and_stashes_normals(monkeypatch):
    from spag4d import pager_model

    H, W = 64, 128

    class _FakePager:
        def get_intrinsics_extrinsics(self, **k): pass
        model = type("M", (), {"to": lambda s, d: s, "eval": lambda s: s})()
        def __call__(self, cubemap, dtype=None, skip_heads=None):
            return {"depth": torch.zeros(1, 6, 1, 8, 8),
                    "normals": torch.zeros(1, 6, 3, 8, 8),
                    "sky": torch.zeros(1, 6, 1, 8, 8),
                    "scale": torch.zeros(1)}
        def process_depth_output(self, *a, **k):
            return torch.full((H // 2, W // 2), 5.0), None
        def process_normals_output(self, *a, **k):
            n = torch.zeros(H // 2, W // 2, 3); n[..., 1] = 1.0
            return n, None

    monkeypatch.setattr(pager_model, "build_pager", lambda *a, **k: _FakePager())
    monkeypatch.setattr(pager_model, "_load_cfg", lambda: type("C", (), {"face_size": 504})())
    monkeypatch.setattr(pager_model, "_erp_to_cubemap",
                        lambda chw, face_size: torch.zeros(6, 3, 8, 8))

    m = pager_model.PaGeRModel(_FakePager(),
                               cfg=type("C", (), {"face_size": 504})(),
                               device=torch.device("cpu"), metric=False)
    image = torch.zeros(H, W, 3, dtype=torch.uint8)
    depth, sky = m.predict(image)

    assert depth.shape == (H, W)                      # upsampled to working res
    assert sky.shape == (H, W) and sky.dtype == torch.bool
    assert m.last_normals.shape == (H, W, 3)          # stashed, world-frame
    assert m.depth_convention == "radial"
    assert m.native_resolution == (H // 2, W // 2)    # recorded pre-upsample
    # normals re-normalized to unit length after bilinear upsample
    norms = m.last_normals.reshape(-1, 3).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pager_model.py -v`
Expected: FAIL with `ModuleNotFoundError: spag4d.pager_model` (or `AttributeError`).

- [ ] **Step 3: Write the wrapper**

```python
# spag4d/pager_model.py
"""PaGeR depth backend wrapper. Same role/contract as DA360Model / DAPModel.

Wraps the vendored unified prs-eth/PaGeR (DA3 cubemap multi-view) model.
predict() returns (depth, sky_mask) to fit the existing (depth, mask) slot;
normals + native_resolution + convention are stashed on the instance for
core.py to pull. Weights are CC BY-NC 4.0 (non-commercial).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .pager_arch import build_pager, _ensure_on_path

PAGER_REPO = "prs-eth/PaGeR"
PAGER_CACHE_DIR = Path.home() / ".cache" / "spag4d" / "pager"


def _load_cfg():
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    return OmegaConf.load(hf_hub_download(PAGER_REPO, "config.yaml",
                                          cache_dir=str(PAGER_CACHE_DIR)))


def _erp_to_cubemap(chw: torch.Tensor, face_size: int) -> torch.Tensor:
    """chw: (3,H,W) in [-1,1]. Returns (6,3,face,face). Signature pinned in Task 2."""
    _ensure_on_path()
    from src.utils.geometry_utils import erp_to_cubemap
    return erp_to_cubemap(chw, face_w=face_size, fov=90.0)


class PaGeRModel:
    def __init__(self, pager, cfg, device: torch.device, metric: bool = False):
        self._pager = pager
        self._cfg = cfg
        self.device = device
        self.metric = metric
        self.depth_convention = "radial"
        self.last_normals: torch.Tensor | None = None
        self.native_resolution: tuple[int, int] | None = None

    @classmethod
    def load(cls, device: torch.device = torch.device("cuda"),
             metric: bool = False) -> "PaGeRModel":
        cfg = _load_cfg()
        pager = build_pager(PAGER_REPO, cfg=cfg, device=device)
        pager.get_intrinsics_extrinsics(image_size=cfg.face_size,
                                        fov=getattr(cfg, "cube_fov", 90.0))
        pager.model.to(device).eval()
        return cls(pager, cfg, device, metric=metric)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """image: (H,W,3) uint8/float on device. Returns (depth, sky_mask)."""
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        H, W = image.shape[:2]

        chw = image.permute(2, 0, 1) * 2.0 - 1.0            # (3,H,W) in [-1,1]
        cubemap = _erp_to_cubemap(chw, self._cfg.face_size).unsqueeze(0).to(self.device)

        skip = {"scale_indoor"} if not self.metric else set()
        pred = self._pager(cubemap, dtype=torch.float16, skip_heads=skip)

        depth_n, _ = self._pager.process_depth_output(
            pred["depth"][0], pred["sky"][0], (H, W), log_scale=pred.get("scale"))
        normals_n, _ = self._pager.process_normals_output(pred["normals"][0], pred["sky"][0], (H, W))
        sky_logits = pred["sky"][0]

        depth_n = torch.as_tensor(depth_n, device=self.device).float()
        normals_n = torch.as_tensor(normals_n, device=self.device).float()
        self.native_resolution = tuple(depth_n.shape[:2])

        # Upsample to working resolution: depth bilinear, sky nearest, normals bilinear+renorm
        depth = F.interpolate(depth_n[None, None], size=(H, W),
                              mode="bilinear", align_corners=False)[0, 0]

        sky_prob = torch.sigmoid(torch.as_tensor(sky_logits, device=self.device).float())
        if sky_prob.dim() == 3:
            sky_prob = sky_prob.mean(0) if sky_prob.shape[0] in (1, 3) else sky_prob
        sky_prob = sky_prob.reshape(1, 1, *sky_prob.shape[-2:])
        sky = F.interpolate(sky_prob, size=(H, W), mode="nearest")[0, 0] > 0.5

        n = normals_n.permute(2, 0, 1)[None]
        n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)[0]
        n = n.permute(1, 2, 0)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.last_normals = n

        return depth, sky.bool()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pager_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spag4d/pager_model.py tests/test_pager_model.py
git commit -m "feat(pager): PaGeRModel wrapper (depth + stashed normals/sky)"
```

---

### Task 4: Register `pager` in the `core.py` dispatch

**Files:**
- Modify: `spag4d/core.py:56-72` (`_get_depth_model`) and `:160-199` (depth path)

- [ ] **Step 1: Add lazy construction in `_get_depth_model`**

Insert a `pager` branch before the `da360` branch in `_get_depth_model`:

```python
        elif name == "pager":
            from .pager_model import PaGeRModel
            model = PaGeRModel.load(device=self.device,
                                    metric=getattr(self, "_pager_metric", False))
```

- [ ] **Step 2: Capture the optional channels in `convert()`**

In `convert()`, after `depth_raw, _ = depth_engine.predict(image_tensor)` (core.py:167), the second return value currently `_` carries the sky mask for PaGeR. Replace that line and add channel capture:

```python
        with torch.inference_mode():
            depth_raw, pred_mask = depth_engine.predict(image_tensor)
        depth = depth_raw * global_scale

        # PaGeR-only optional channels (None for da360/dap)
        pager_sky = None
        pager_normals = None
        if dm_name == "pager":
            print(f"[SPAG4D] PaGeR native depth {depth_engine.native_resolution} "
                  f"-> upsampled to {tuple(depth.shape)}")
            if getattr(self, "_pager_use_sky", False):
                pager_sky = pred_mask.cpu().numpy() if pred_mask is not None else None
            if getattr(self, "_pager_use_normals", False):
                n = depth_engine.last_normals
                pager_normals = n.cpu().numpy() if n is not None else None
```

(`dm_name` is already computed at core.py:162 as `depth_model or self.default_depth_model`.)

- [ ] **Step 3: Add the pager option fields to `__init__`**

In `SPAG4D.__init__`, after `self.generator = generator or depth_model` (core.py:50), add:

```python
        self._pager_metric = False
        self._pager_use_sky = False
        self._pager_use_normals = False
```

These are set by `convert()` from kwargs in Task 5 / Task 7 / Task 9. (`pager_sky`/`pager_normals` stay `None` until those tasks pass them downstream — for now this task only proves the `pager` generator loads and produces depth.)

- [ ] **Step 4: Manual dispatch check (guarded — needs weights)**

Run: `.venv/Scripts/python.exe -m spag4d convert tests/data/<pano>.jpg /tmp/pager_out.ply --generator pager --stride 4`
Expected: runs end-to-end, prints `PaGeR native depth ... -> upsampled to ...`, writes a PLY. (Skip if weights/GPU unavailable in CI; this is a local check.)

- [ ] **Step 5: Run the existing test suite (regression guard)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py tests/test_scene_filter.py tests/test_pager_model.py -v`
Expected: all PASS (da360/dap paths untouched).

- [ ] **Step 6: Commit**

```bash
git add spag4d/core.py
git commit -m "feat(pager): register pager generator in core dispatch"
```

---

### Task 5: CLI wiring + download-models + NC warning

**Files:**
- Modify: `spag4d/cli.py:40-41` (`--generator` choice), `:46-67` (convert signature), `:92-116` (construction/call), `:148-150` (`download-models` choice), `:187+` (download body)
- Modify: `spag4d/core.py` `convert()` signature (accept `pager_metric`)

- [ ] **Step 1: Extend the `--generator` choice and add pager flags**

Change line 40 and add three options:

```python
@click.option('--generator', type=click.Choice(['da360', 'dap', 'sharp360', 'pager']),
              default=None, help='Generator mode: da360, dap, sharp360, or pager (overrides --depth-model)')
@click.option('--pager-metric', is_flag=True,
              help='PaGeR: use the metric scale head (default scale-invariant depth)')
@click.option('--pager-use-sky', is_flag=True,
              help='PaGeR: use the learned sky mask for depth-range fit (Phase 2)')
@click.option('--pager-use-normals', is_flag=True,
              help='PaGeR: use surface normals for grazing-angle clip (Phase 3)')
```

Add the matching params to the `convert(...)` function signature: `pager_metric: bool, pager_use_sky: bool, pager_use_normals: bool`.

- [ ] **Step 2: Print the non-commercial warning**

In the `if not quiet:` block (after line 90), add:

```python
        if generator == 'pager':
            click.echo("  NOTE: PaGeR weights are CC BY-NC 4.0 — non-commercial / evaluation use only.")
```

- [ ] **Step 3: Pass the flags through the convert call**

Add to the `converter.convert(...)` kwargs in `run_single` (after line 115):

```python
            pager_metric=pager_metric,
            pager_use_sky=pager_use_sky,
            pager_use_normals=pager_use_normals,
```

- [ ] **Step 4: Accept the kwargs in `core.convert()` and set instance flags**

Add `pager_metric: bool = False, pager_use_sky: bool = False, pager_use_normals: bool = False` to `convert()`'s signature, and at the top of `convert()` (before depth estimation) set:

```python
        self._pager_metric = pager_metric
        self._pager_use_sky = pager_use_sky
        self._pager_use_normals = pager_use_normals
```

- [ ] **Step 5: Add pager to download-models**

Change line 149 choice to include `'pager'`, then add a download branch:

```python
    if model in ('pager', 'all'):
        try:
            from huggingface_hub import snapshot_download
            from spag4d.pager_model import PAGER_CACHE_DIR
            click.echo("Downloading PaGeR weights (prs-eth/PaGeR, ~5.7GB, CC BY-NC 4.0)...")
            path = snapshot_download("prs-eth/PaGeR", cache_dir=str(PAGER_CACHE_DIR))
            click.echo(f"PaGeR weights cached at: {path}")
        except Exception as e:
            click.echo(f"PaGeR download failed: {e}", err=True)
            if model == 'pager':
                raise click.Abort()
```

- [ ] **Step 6: Verify CLI help shows the new flags**

Run: `.venv/Scripts/python.exe -m spag4d convert --help`
Expected: `--generator` lists `pager`; `--pager-metric`, `--pager-use-sky`, `--pager-use-normals` appear.

- [ ] **Step 7: Commit**

```bash
git add spag4d/cli.py spag4d/core.py
git commit -m "feat(pager): CLI flags, NC warning, download-models --model pager"
```

---

## Phase 2 — Sky mask

### Task 6: `compute_scene_defaults` accepts a sky mask

**Files:**
- Modify: `spag4d/scene_analysis.py:8-55`
- Test: `tests/test_scene_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scene_analysis.py
def test_sky_mask_excludes_sky_from_depth_range():
    from spag4d.scene_analysis import compute_scene_defaults
    # Ground 2-8m on the bottom half; "sky" = 500m on the top half.
    depth = np.empty((100, 200), dtype=np.float32)
    depth[:50] = 500.0
    depth[50:] = np.random.uniform(2.0, 8.0, size=(50, 200))
    sky = np.zeros((100, 200), dtype=bool)
    sky[:50] = True

    masked = compute_scene_defaults(depth, sky_mask=sky)
    unmasked = compute_scene_defaults(depth)  # sky drags the range up

    assert masked["depth_max"] < 20.0, "sky-excluded depth_max should track the ground"
    assert unmasked["depth_max"] > masked["depth_max"]


def test_sky_mask_none_is_backward_compatible():
    from spag4d.scene_analysis import compute_scene_defaults
    depth = np.random.uniform(0.5, 5.0, size=(128, 256)).astype(np.float32)
    assert compute_scene_defaults(depth) == compute_scene_defaults(depth, sky_mask=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py::test_sky_mask_excludes_sky_from_depth_range -v`
Expected: FAIL with `TypeError: compute_scene_defaults() got an unexpected keyword argument 'sky_mask'`.

- [ ] **Step 3: Add the `sky_mask` parameter**

Change the signature (line 8-11) to add `sky_mask`, and intersect it into `valid`:

```python
def compute_scene_defaults(
    depth_map: np.ndarray,
    image_height: int | None = None,
    sky_mask: np.ndarray | None = None,
) -> dict:
```

Then change line 31 from `valid = (depth_map > 0.01) & np.isfinite(depth_map)` to:

```python
    valid = (depth_map > 0.01) & np.isfinite(depth_map)
    if sky_mask is not None:
        valid = valid & ~sky_mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py -v`
Expected: all PASS (old tests + two new).

- [ ] **Step 5: Commit**

```bash
git add spag4d/scene_analysis.py tests/test_scene_analysis.py
git commit -m "feat(pager): sky_mask param in compute_scene_defaults"
```

---

### Task 7: Thread the sky mask through the pager branch

**Files:**
- Modify: `spag4d/core.py:172-181` (scene defaults call)

- [ ] **Step 1: Pass `pager_sky` into `compute_scene_defaults`**

Change the scene-defaults call (core.py:174) from
`scene_defaults = compute_scene_defaults(depth_np, image_height=H)` to:

```python
        scene_defaults = compute_scene_defaults(depth_np, image_height=H, sky_mask=pager_sky)
```

(`pager_sky` is `None` for every non-pager generator and when `--pager-use-sky` is off, so behavior is unchanged elsewhere.)

- [ ] **Step 2: Manual check with and without the flag (guarded — needs weights)**

Run both:
`.venv/Scripts/python.exe -m spag4d convert tests/data/<outdoor>.jpg /tmp/no_sky.ply --generator pager --stride 4`
`.venv/Scripts/python.exe -m spag4d convert tests/data/<outdoor>.jpg /tmp/sky.ply --generator pager --pager-use-sky --stride 4`
Expected: the `--pager-use-sky` run logs a tighter `depth=[...]` range (sky no longer dragging `depth_max`).

- [ ] **Step 3: Regression guard**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_analysis.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add spag4d/core.py
git commit -m "feat(pager): wire learned sky mask into scene-range fit"
```

---

## Phase 3 — Normals

### Task 8: `prune_grazing_angle` accepts world-frame normals

**Files:**
- Modify: `spag4d/scene_filter.py:368-447`
- Test: `tests/test_scene_filter.py`

- [ ] **Step 1: Write the failing test**

The converter ray frame is `view_ray = means / ||means||`. A surface facing the camera has its normal antiparallel to the ray (`|dot| ≈ 1`); an edge-on (grazing) surface has the normal perpendicular to the ray (`|dot| ≈ 0`). With normals supplied, keep iff `|dot(view_ray, normal)| > cos(max_angle_deg)`.

```python
# append to tests/test_scene_filter.py
class TestGrazingAngleNormals:
    def test_face_on_kept_edge_on_removed(self):
        from spag4d.scene_filter import prune_grazing_angle
        H, W = 64, 128
        # Two gaussians on the +x axis (theta=pi/2 wrap aside): use simple positions.
        means = torch.tensor([[5.0, 0.0, 0.0],     # view_ray = +x
                              [0.0, 5.0, 0.0]],     # view_ray = +y
                             dtype=torch.float32)
        gaussians = {"means": means,
                     "scales": torch.ones(2, 3),
                     "quats": torch.zeros(2, 4),
                     "opacities": torch.zeros(2),
                     "colors": torch.zeros(2, 3)}
        depth = np.full((H, W), 5.0, dtype=np.float32)
        # Normal map: make the FIRST gaussian face-on (normal = +x at its pixel),
        # the SECOND edge-on (normal = +x while its ray is +y -> dot 0).
        normals = np.zeros((H, W, 3), dtype=np.float32)
        normals[..., 0] = 1.0  # everywhere +x

        out = prune_grazing_angle(gaussians, depth, stride=1,
                                  max_angle_deg=60.0, normals=normals)
        # gaussian 0 (ray +x, normal +x -> |dot|=1) kept; gaussian 1 (|dot|=0) removed
        assert out["means"].shape[0] == 1
        assert torch.allclose(out["means"][0], means[0])

    def test_normals_none_falls_back(self):
        from spag4d.scene_filter import prune_grazing_angle
        H, W = 64, 128
        means = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)
        gaussians = {"means": means, "scales": torch.ones(1, 3),
                     "quats": torch.zeros(1, 4), "opacities": torch.zeros(1),
                     "colors": torch.zeros(1, 3)}
        depth = np.full((H, W), 5.0, dtype=np.float32)
        # no normals -> depth-gradient path; flat depth -> nothing pruned
        out = prune_grazing_angle(gaussians, depth, stride=1, max_angle_deg=60.0)
        assert out["means"].shape[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_filter.py::TestGrazingAngleNormals -v`
Expected: FAIL (`unexpected keyword argument 'normals'`).

- [ ] **Step 3: Add the `normals` branch**

Add `normals: np.ndarray | None = None` to the signature (after `max_angle_deg`). Then, after the existing pixel back-projection that computes `px_row`, `px_col` (scene_filter.py:430-431) and before `sampled_grad` is used (line 434), insert a normals fast-path:

```python
    means_np = gaussians['means'].detach().cpu().numpy()
    if normals is not None:
        r_n = np.linalg.norm(means_np, axis=1, keepdims=True)
        view_ray = means_np / np.maximum(r_n, 1e-8)           # == rhat, converter frame
        sampled_normal = normals[px_row, px_col]              # (N,3) world-frame
        dot = np.abs(np.sum(view_ray * sampled_normal, axis=1))
        keep_mask_np = dot > np.cos(np.radians(max_angle_deg))
    else:
        sampled_grad = relative_grad[px_row, px_col]
        keep_mask_np = sampled_grad < max_relative_grad
```

Remove the now-duplicated original `sampled_grad`/`keep_mask_np` lines (434-436) so only the `else` branch computes the gradient path. Keep everything after `keep_mask = torch.from_numpy(...)` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_filter.py -v`
Expected: all PASS (existing sky/pole tests + two new).

- [ ] **Step 5: Commit**

```bash
git add spag4d/scene_filter.py tests/test_scene_filter.py
git commit -m "feat(pager): normal-aware grazing-angle clip with depth-gradient fallback"
```

---

### Task 9: Thread normals through the pager branch + world-frame check

**Files:**
- Modify: `spag4d/core.py:211-220` (grazing-angle filter call)

- [ ] **Step 1: Pass `pager_normals` into `prune_grazing_angle`**

In the grazing-angle filter block (core.py:215), change the call to forward normals:

```python
                gaussians = prune_grazing_angle(
                    gaussians, depth_np, stride=stride, max_angle_deg=grazing_angle,
                    normals=pager_normals,
                )
```

(`pager_normals` is `None` for non-pager generators and when `--pager-use-normals` is off.)

- [ ] **Step 2: World-frame sanity check (guarded — needs weights)**

Add a one-off normals visualization to confirm PaGeR's world frame matches the converter frame before trusting the clip. Run:
`.venv/Scripts/python.exe scripts/pager_smoke.py tests/data/<interior>.jpg`
then extend the smoke script to save `((normals * 0.5 + 0.5) * 255)` as a PNG and eyeball it: walls/floor/ceiling must read as coherent flat color regions. If the normals look axis-swapped relative to the converter's `[sin φ cos θ, cos φ, -sin φ sin θ]` frame, add the corresponding axis permutation/sign flip inside `PaGeRModel.predict` (Task 3) where `self.last_normals` is assigned, and note it in the docstring.

- [ ] **Step 3: A/B the grazing clip (guarded — needs weights)**

Run both and compare splat counts / visual striation behind objects:
`.venv/Scripts/python.exe -m spag4d convert tests/data/<pano>.jpg /tmp/no_n.ply --generator pager --stride 2`
`.venv/Scripts/python.exe -m spag4d convert tests/data/<pano>.jpg /tmp/n.ply --generator pager --pager-use-normals --stride 2`
Expected: the normals run removes the same or fewer *correct* surfaces vs. the depth-gradient path, with cleaner edges.

- [ ] **Step 4: Regression guard**

Run: `.venv/Scripts/python.exe -m pytest tests/test_scene_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spag4d/core.py scripts/pager_smoke.py
git commit -m "feat(pager): wire world-frame normals into grazing-angle clip"
```

---

## Phase 4 — Polish

### Task 10: Docs, provenance, regression guard, and full smoke test

**Files:**
- Modify: `README.md` (generator table + license table)
- Modify: `INSTALL.md` (optional requirements + WSL2 fallback note)
- Modify: `spag4d/ply_writer.py` (optional header comment — only if a comment hook exists)
- Test: `tests/test_pager_model.py` (add a weights-guarded integration smoke test)

- [ ] **Step 1: README — add the fourth generator and license rows**

Add a `pager` row to the generators/modes table: "PaGeR (DA3 cubemap) — strongest outdoor depth + learned sky + world normals; non-commercial weights." Add a fourth row to the license table: "PaGeR weights — CC BY-NC 4.0 — non-commercial / evaluation only (inherited from DA3 ViT-Giant backbone)." Note the vendored commit hash from Task 1.

- [ ] **Step 2: INSTALL.md — optional deps + fallback**

Add: "Optional PaGeR backend: `pip install -r requirements-pager.txt`, then `python -m spag4d download-models --model pager`. Native Windows is expected to work (pure-PyTorch); if `open_clip`/`pytorch360convert`/DINOv2 fail, run PaGeR under WSL2 (see spec §3.2)."

- [ ] **Step 2b: PLY provenance (only if a header-comment hook exists)**

Check `spag4d/ply_writer.py` for an existing comment/metadata hook. If present, stash `generator=pager` + `metric=<bool>` in the header. If `save_ply_gsplat` has no comment parameter, SKIP this step (do not add a new parameter just for provenance — out of scope per spec §9).

- [ ] **Step 3: Add a weights-guarded integration smoke test**

```python
# append to tests/test_pager_model.py
import os
from pathlib import Path

def _weights_present():
    from spag4d.pager_model import PAGER_CACHE_DIR
    return Path(PAGER_CACHE_DIR).exists() and any(Path(PAGER_CACHE_DIR).rglob("*.safetensors"))

@pytest.mark.skipif(not _weights_present(), reason="PaGeR weights not downloaded")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pager_end_to_end_depth_shape():
    from spag4d.pager_model import PaGeRModel
    data = sorted(Path("tests/data").glob("*.jpg"))
    if not data:
        pytest.skip("no test panorama in tests/data")
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(data[0]).convert("RGB"))
    H, W = img.shape[:2]
    m = PaGeRModel.load(device=torch.device("cuda"))
    depth, sky = m.predict(torch.from_numpy(img).cuda())
    assert depth.shape == (H, W)
    assert sky.shape == (H, W) and sky.dtype == torch.bool
    assert m.last_normals.shape == (H, W, 3)
    assert m.native_resolution[0] < H or m.native_resolution[0] <= 1100  # ~1024 native
```

- [ ] **Step 4: Full regression guard**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: all PASS or SKIP (PaGeR integration test SKIPs without weights/GPU; da360/dap/scene tests PASS).

- [ ] **Step 5: Confirm DA360/DAP output is unchanged (additive guarantee)**

Run a DA360 conversion that you can compare against a pre-integration baseline (same input, same flags):
`.venv/Scripts/python.exe -m spag4d convert tests/data/<pano>.jpg /tmp/da360_after.ply --generator da360 --stride 4`
Expected: identical splat count and depth range to a baseline run on `main` (the pager branch never executes for `da360`).

- [ ] **Step 6: Commit**

```bash
git add README.md INSTALL.md tests/test_pager_model.py spag4d/ply_writer.py
git commit -m "docs(pager): README/INSTALL rows, provenance, weights-guarded smoke test"
```

---

## Self-Review

**Spec coverage:**
- §5.1 vendoring → Task 1 ✅
- §5.2 `PaGeRModel` (predict→(depth,sky), normals stashed, upsample rules, native_resolution) → Task 3 ✅
- §5.3 core dispatch → Tasks 4, 7, 9 ✅
- §5.4 `compute_scene_defaults(sky_mask)` → Task 6, wired Task 7 ✅
- §5.5 `prune_grazing_angle(normals)` + world-frame check → Tasks 8, 9 ✅
- §5.6 CLI flags + NC warning → Task 5 ✅
- §5.7 weights/deps/README/INSTALL → Tasks 1, 5, 10 ✅
- §2 residual unknowns (Windows, erp_to_cubemap sig, radial) → Task 2 smoke gate ✅
- §7 validation (regression guard, hole-rate via camera_rig) → Tasks 4/10 regression guards; hole-rate A/B noted in Tasks 4/9 manual checks. (Hole-rate measurement reuses `refine/camera_rig.py` as an evaluation step, not new code — left to the A/B runs.)

**Placeholder scan:** Manual-check steps that need the 5.7GB weights/GPU are explicitly marked "(guarded — needs weights)" with a concrete command; they are validation, not unfinished code. All code steps contain complete code.

**Type consistency:** `PaGeRModel.load()`, `.predict()→(depth, sky)`, `.last_normals`, `.native_resolution`, `.depth_convention`, `PAGER_REPO`, `PAGER_CACHE_DIR`, `build_pager`, `_ensure_on_path`, `_erp_to_cubemap`, `_load_cfg` are used consistently across Tasks 1/3/4/5/10. Channel names `pager_sky`/`pager_normals` and instance flags `_pager_metric`/`_pager_use_sky`/`_pager_use_normals` consistent across Tasks 4/5/7/9.

**Known soft spots flagged for the implementer:** exact `erp_to_cubemap` signature (Task 2 Step 2 pins it); the `process_depth_output`/`process_normals_output` argument order (confirm against `src/pager.py` during Task 2 — adjust Task 3 if upstream differs); normals axis convention (Task 9 Step 2 verifies and corrects in Task 3's assignment site).
