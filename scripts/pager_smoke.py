"""Phase 1 smoke gate for the PaGeR backend.

Confirms the vendored PaGeR runs end-to-end through PaGeRModel on native Windows,
produces radial ERP depth at working resolution, and emits a coherent world-frame
normal map + sky mask. Run AFTER downloading weights:

    python -m spag4d download-models --model pager
    .venv/Scripts/python.exe scripts/pager_smoke.py <pano.jpg> [--metric]

Writes <pano>_depth.png / _normals.png / _sky.png next to the input so you can
eyeball: depth = straight-line (radial) distance, normals = flat coherent walls/
floor/ceiling (Task 9 world-frame check), sky = clean horizon segmentation.

GATE: if this crashes on native Windows, switch PaGeR to the WSL2 fallback
(see docs/superpowers/specs/2026-06-01-pager-integration-design.md §3.2) before
wiring anything else. (Import + signatures are already verified; this exercises
the GPU forward pass + geometry stitch.)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _save_gray(arr: np.ndarray, path: Path) -> None:
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pano")
    ap.add_argument("--metric", action="store_true", help="use the metric scale head")
    args = ap.parse_args()

    from spag4d.pager_model import PaGeRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pano = Path(args.pano)
    img = np.array(Image.open(pano).convert("RGB"))
    H, W = img.shape[:2]
    print(f"input: {pano.name}  {W}x{H}  device={device}  metric={args.metric}")

    model = PaGeRModel.load(device=device, metric=args.metric)
    depth, sky = model.predict(torch.from_numpy(img).to(device))
    normals = model.last_normals

    d = depth.detach().float().cpu().numpy()
    n = normals.detach().float().cpu().numpy()
    s = sky.detach().cpu().numpy()

    print(f"native_resolution (effective): {model.native_resolution}")
    print(f"depth shape {d.shape}  convention={model.depth_convention}")
    print(f"depth min/median/max: {np.nanmin(d):.2f} / {np.nanmedian(d):.2f} / {np.nanmax(d):.2f}")
    print(f"sky coverage: {100.0 * s.mean():.1f}%   normals shape {n.shape}")
    print("RADIAL CHECK: pick a feature at a known distance; confirm depth ~= straight-line distance.")

    _save_gray(d, pano.with_name(pano.stem + "_depth.png"))
    Image.fromarray(((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).save(
        pano.with_name(pano.stem + "_normals.png"))
    _save_gray(s.astype(np.float32), pano.with_name(pano.stem + "_sky.png"))
    print(f"wrote {pano.stem}_depth.png / _normals.png / _sky.png")


if __name__ == "__main__":
    main()
