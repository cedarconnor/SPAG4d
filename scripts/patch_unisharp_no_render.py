"""Idempotently patch UniSHARP's scripts/infer_unisharp.py to add --no-render.

The `--no-render` flag skips UniSHARP's gsplat GIF/preview rendering so inference
produces only the PLY (+ metadata). This avoids gsplat's CUDA JIT build entirely,
which does not compile cleanly against torch 2.8 on native Windows. SPAG4d only
consumes the PLY, so rendering is pure overhead for this integration.

Usage:
    python scripts/patch_unisharp_no_render.py D:/repos/UniSHARP

Safe to run repeatedly: it detects an already-patched file and exits cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch(repo_dir: str) -> int:
    script = Path(repo_dir) / "scripts" / "infer_unisharp.py"
    if not script.exists():
        print(f"[ERROR] not found: {script}")
        return 1

    text = script.read_text(encoding="utf-8")
    if "no_render" in text or "no-render" in text:
        print("[OK] infer_unisharp.py already patched for --no-render.")
        return 0

    # 1. Add the argparse flag right after --save-ply.
    needle = 'p.add_argument("--save-ply", action="store_true")'
    if needle not in text:
        print("[ERROR] could not find the --save-ply argument to anchor the patch.")
        return 1
    text = text.replace(
        needle,
        needle
        + "\n    p.add_argument(\n"
        '        "--no-render",\n'
        '        action="store_true",\n'
        '        help="Skip GIF/preview rendering (gsplat). PLY-only; Windows-friendly.",\n'
        "    )",
        1,
    )

    # 2. Gate the per-pose render loops (3 camera branches share identical lines).
    text = text.replace(
        "        for pose in forward_poses:",
        "        for pose in ([] if args.no_render else forward_poses):",
    )
    text = text.replace(
        "        for pose in rotate_poses:",
        "        for pose in ([] if args.no_render else rotate_poses):",
    )

    # 3. Gate the GIF writes.
    gif_block = (
        '    _save_gif(forward_frames, sample_dir / "forward.gif", duration_ms=GIF_DURATION_MS)\n'
        '    _save_gif(rotate_frames, sample_dir / "rotate.gif", duration_ms=GIF_DURATION_MS)'
    )
    if gif_block in text:
        text = text.replace(
            gif_block,
            "    if not args.no_render:\n"
            '        _save_gif(forward_frames, sample_dir / "forward.gif", duration_ms=GIF_DURATION_MS)\n'
            '        _save_gif(rotate_frames, sample_dir / "rotate.gif", duration_ms=GIF_DURATION_MS)',
            1,
        )

    script.write_text(text, encoding="utf-8")
    print(f"[OK] patched {script} for --no-render.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python patch_unisharp_no_render.py <UniSHARP repo dir>")
        raise SystemExit(2)
    raise SystemExit(patch(sys.argv[1]))
