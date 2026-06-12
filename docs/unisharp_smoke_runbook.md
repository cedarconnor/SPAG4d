# UniSHARP Backend — Smoke & Viewer Reality-Check Runbook (M1.5)

This is the **manual, GPU-gated** validation step for the UniSHARP `sharp360`
backend. The code (Tasks 1–9 of the integration plan) is fully unit-tested
without a GPU; this runbook confirms the backend against a real UniSHARP install
and a real panorama, and resolves the one unverified detail — how UniSHARP
actually encodes `color_space` in its PLY.

Run it once before relying on the backend in production.

---

## 1. One-time environment setup

UniSHARP runs **out-of-process** in its own conda env (Python 3.12 / torch 2.8).
Never import it into SPAG4d's process.

```bash
git clone https://github.com/Insta360-Research-Team/UniSHARP.git
cd UniSHARP
conda create -n unisharp python=3.12 -y
conda activate unisharp
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
git clone https://github.com/lpiccinelli-eth/UniK3D.git UniK3D
# 3dgeer is only needed for fisheye rendering — skip for the panorama path.
# Download a checkpoint from HuggingFace Insta360-Research/Unisharp -> step_XXXXXXX.pt
```

Record the three paths you'll reuse:

| Var | Example |
|-----|---------|
| `SPAG4D_UNISHARP_REPO` | `D:\repos\UniSHARP` |
| `SPAG4D_UNISHARP_PYTHON` | `D:\envs\unisharp\python.exe` (the conda env's python) |
| `SPAG4D_UNISHARP_CHECKPOINT` | `D:\models\unisharp\step_0100000.pt` |

---

## 2. Smoke test (copy mode + debug artifacts)

Use a real 2:1 ERP panorama. From the SPAG4d repo root (Windows PowerShell):

```powershell
.venv\Scripts\python.exe -m spag4d convert test_pano.jpg test_unisharp.ply `
  --generator unisharp360 `
  --unisharp-repo D:\repos\UniSHARP `
  --unisharp-python D:\envs\unisharp\python.exe `
  --unisharp-checkpoint D:\models\unisharp\step_0100000.pt `
  --unisharp-save-debug
```

**Expected:**
- command completes with exit code 0
- `test_unisharp.ply` exists with > 0 vertices
- `test_unisharp_unisharp_debug/` contains `raw_gaussians.ply`, `metadata.json`,
  and `forward.gif` + `rotate.gif`

If you instead see `UniSHARP completed but no gaussians.ply` → the `--save-ply`
flag did not reach the subprocess (regression in the adapter). If you see a
subprocess traceback, the full UniSHARP stdout/stderr is included in the error.

---

## 3. Confirm the `color_space` PLY encoding  ← the one unverified detail

Our fixtures assume `color_space` is a PLY **comment**. The real writer may use a
supplement **element** instead. Inspect the raw header:

```powershell
.venv\Scripts\python.exe -c "from plyfile import PlyData; p=PlyData.read(r'test_unisharp_unisharp_debug\raw_gaussians.ply'); print('comments:', p.comments); print('elements:', [e.name for e in p.elements]); [print(e.name, [pr.name for pr in e.properties]) for e in p.elements]"
```

Decide based on what you see:

- **`color_space` is a comment** (e.g. `comment color_space sRGB`) or **absent** →
  no change needed; `spag4d/unisharp_format.py::_read_color_space` already handles it.
- **`color_space` is a supplement element** → update `_read_color_space` in
  `spag4d/unisharp_format.py` to read it from that element, add a regression test
  to `tests/test_unisharp_format.py` that builds the fixture that way, and commit.

Either way, `convert` mode preserves `f_dc` colors as-is, so M3 correctness does
not depend on reading `color_space` — this step only future-proofs scale/color work.

---

## 4. Viewer reality check (kills-or-confirms `convert` mode)

Open the **copy-mode** output `test_unisharp.ply` in:
1. **SuperSplat** (https://supersplat.dev)
2. SPAG4d's own GaussianSplats3D viewer (`python -m spag4d serve` → load the PLY)

Check: orientation (is up actually up?), scale (plausible room size?), color
(natural, not washed-out/over-saturated), opacity (solid surfaces, not ghostly).

**Decision:**
- Supplement elements load fine everywhere and orientation/color are acceptable →
  `copy` mode is the default; `convert` mode is an optional fallback.
- A strict loader chokes on the supplement elements → recommend
  `--unisharp-format-mode convert` for that loader (re-run §2 with that flag and
  re-verify).

Record the outcome here:

```
Date run:            ____________
UniSHARP checkpoint: ____________
Pano used:           ____________
Vertices produced:   ____________
color_space encoding: comment | element | absent   (circle one)
SuperSplat load:     ok | needs-convert
SPAG4d viewer load:  ok | needs-convert
Orientation / scale / color notes:
  ____________________________________________________________
Chosen default format-mode: copy | convert
```

---

## 5. Comparison pass (optional, informs the future hybrid plan / M6)

Run the same panorama through each generator and note differences. This data
scopes the eventual hybrid backend (UniSHARP as pole/seam filler).

```powershell
.venv\Scripts\python.exe -m spag4d convert test_pano.jpg out_da360.ply  --generator da360
.venv\Scripts\python.exe -m spag4d convert test_pano.jpg out_sharp.ply  --generator sharp360 --sharp-backend sharp
.venv\Scripts\python.exe -m spag4d convert test_pano.jpg out_uni.ply    --generator unisharp360 `
  --unisharp-repo D:\repos\UniSHARP --unisharp-python D:\envs\unisharp\python.exe `
  --unisharp-checkpoint D:\models\unisharp\step_0100000.pt
```

Compare: gaussian count, load success, seam behavior (cubemap-SHARP shows face
seams; UniSHARP native-ERP should not), pole (nadir/zenith) quality, and
close-object disocclusion. UniSHARP's expected edge is cleaner poles/seams — not
a larger camera-travel envelope (its novel-view baseline is 0.2 m forward /
0.1 m rotate radius).

```
da360   gaussians: ______   seams: ______   poles: ______
sharp   gaussians: ______   seams: ______   poles: ______
uni     gaussians: ______   seams: ______   poles: ______
Notes for hybrid (M6): _______________________________________
```
