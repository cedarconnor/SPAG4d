"""A/B harness: compare refine backends on one scene.

Runs the OmniRoam (native augment) and/or ArtiFixer3D (Docker rebuild) backends on
the same SPAG-4D cloud and prints a comparison table (initial vs final hole
fraction, anchor PSNR, wall-clock). Promote ArtiFixer3D in the docs only if it
wins on real content.

Usage (host venv):
    .venv/Scripts/python.exe experiments/artifixer_eval/ab_compare.py \
        --cloud experiments/artifixer_eval/scene/bell_tower.ply \
        --panorama path/to/pano.jpg --depth path/to/depth.npy \
        --backends artifixer3d,omniroam --out work/ab

OmniRoam needs --panorama + --depth; ArtiFixer3D needs only --cloud.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _run_artifixer3d(cloud, out_dir):
    from spag4d.refine import ArtiFixer3DConfig, refine_splat_artifixer3d
    from spag4d.refine.artifixer3d_pipeline import validate_artifixer_result

    cfg = ArtiFixer3DConfig(enabled=True)
    cfg.work_dir = str(out_dir / "artifixer3d_work")
    out_ply = out_dir / "artifixer3d_refined.ply"
    t0 = time.time()
    res = refine_splat_artifixer3d(str(cloud), cfg, str(out_ply))
    wall = time.time() - t0
    try:
        guard = validate_artifixer_result(cloud, out_ply, cfg)
        anchor_psnr = guard["anchor_psnr"]
    except Exception as e:  # noqa: BLE001 — diagnostic harness
        anchor_psnr = float("nan")
        print(f"[artifixer3d] guard skipped: {e!r}")
    return {
        "initial": res["initial_hole_fraction"],
        "final": res["final_hole_fraction"],
        "anchor_psnr": anchor_psnr,
        "count": res["gaussians_count"],
        "wall_s": wall,
    }


def _run_omniroam(cloud, panorama, depth, out_dir):
    import numpy as np

    from spag4d.refine import OmniRoamConfig, refine_splat_v2

    out_ply = out_dir / "omniroam_refined.ply"
    cfg = OmniRoamConfig(enabled=True)
    t0 = time.time()
    res = refine_splat_v2(
        ply_path=str(cloud), panorama_path=str(panorama),
        depth_map=np.load(str(depth)), config=cfg,
        output_path=str(out_ply), progress_callback=None,
        diagnostics_dir=str(out_dir / "omniroam_diag"),
    )
    wall = time.time() - t0
    return {
        "initial": res["initial_hole_fraction"],
        "final": res["final_hole_fraction"],
        "anchor_psnr": res.get("source_anchor_psnr", float("nan")),
        "count": res["gaussians_count"],
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--panorama", default=None)
    ap.add_argument("--depth", default=None)
    ap.add_argument("--backends", default="artifixer3d,omniroam")
    ap.add_argument("--out", default="work/ab")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    rows = {}
    if "artifixer3d" in backends:
        rows["ArtiFixer3D"] = _run_artifixer3d(args.cloud, out_dir)
    if "omniroam" in backends:
        if not (args.panorama and args.depth):
            print("[omniroam] skipped: --panorama and --depth required")
        else:
            rows["OmniRoam"] = _run_omniroam(args.cloud, args.panorama, args.depth, out_dir)

    print("\n=== refine backend A/B ===")
    print(f"{'backend':<14}{'init hole':>10}{'final hole':>12}{'anchor PSNR':>13}{'count':>11}{'wall (s)':>10}")
    for name, r in rows.items():
        print(f"{name:<14}{r['initial']:>10.3f}{r['final']:>12.3f}"
              f"{r['anchor_psnr']:>13.2f}{r['count']:>11d}{r['wall_s']:>10.1f}")


if __name__ == "__main__":
    main()
