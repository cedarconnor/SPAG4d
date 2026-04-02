"""Diagnostic: render novel views from PLY, detect holes, save comparison images.

Usage:
    python tests/diagnostic_refine_render.py [--ply PATH] [--panorama PATH] [--depth PATH]

Produces output/diagnostics/ with:
  - novel_view_XX.png: rendered from each camera
  - hole_mask_XX.png: detected holes (white = hole)
  - cubemap_XX.png: cubemap face extractions
  - summary.txt: statistics
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spag4d.refine.camera_rig import (
    CameraPose, generate_camera_rig, render_with_hole_mask,
    select_repair_cameras, extract_cubemap_views, _camera_to_RT,
)
from spag4d.refine.format_compat import load_gaussians_from_ply
from spag4d.refine.config import RefineConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def save_rgb(arr, path):
    """Save (H,W,3) float32 [0,1] as PNG."""
    img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
    img.save(path)


def save_mask(arr, path):
    """Save (H,W) float32 [0,1] as grayscale PNG."""
    img = Image.fromarray((arr.clip(0, 1) * 255).astype(np.uint8))
    img.save(path)


def save_side_by_side(left, right, path, labels=None):
    """Save two (H,W,3) images side by side with optional labels."""
    h, w = left.shape[:2]
    gap = 4
    canvas = np.ones((h, w * 2 + gap, 3), dtype=np.float32)
    canvas[:, :w] = left.clip(0, 1)
    canvas[:, w + gap:] = right.clip(0, 1)
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    if labels:
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), labels[0], fill=(255, 255, 0))
            draw.text((w + gap + 10, 10), labels[1], fill=(255, 255, 0))
        except Exception:
            pass
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description="Refine pipeline diagnostic renderer")
    parser.add_argument("--ply", default="output/jobs/4a784dc0-b200-4920-ab72-4f5adbbff603_output.ply")
    parser.add_argument("--panorama", default="output/jobs/4a784dc0-b200-4920-ab72-4f5adbbff603_input.png")
    parser.add_argument("--depth", default="output/jobs/4a784dc0-b200-4920-ab72-4f5adbbff603_depth.npy")
    parser.add_argument("--outdir", default="output/diagnostics")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-views", type=int, default=12,
                        help="Number of azimuthal directions for novel views")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = RefineConfig()

    # --- Load data ---
    logger.info("Loading PLY: %s", args.ply)
    t0 = time.time()
    gaussians = load_gaussians_from_ply(args.ply, device="cuda")
    n_gaussians = gaussians.get_xyz.shape[0]
    logger.info("Loaded %d Gaussians in %.1fs", n_gaussians, time.time() - t0)

    logger.info("Loading panorama: %s", args.panorama)
    panorama = np.array(Image.open(args.panorama)).astype(np.float32) / 255.0
    logger.info("Panorama shape: %s", panorama.shape)

    logger.info("Loading depth: %s", args.depth)
    depth_map = np.load(args.depth)
    logger.info("Depth shape: %s, range: [%.2f, %.2f]",
                depth_map.shape, depth_map.min(), depth_map.max())

    # --- Phase 1a: Generate novel-view cameras ---
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth_map,
        num_directions=args.num_views,
        num_depths=3,
        fov_deg=config.camera_fov,
        translation_fracs=config.translation_fracs,
        resolution=args.resolution,
    )
    logger.info("Generated %d cameras", len(cameras))

    # --- Phase 1b: Render novel views and detect holes ---
    renders = []
    masks = []
    hole_fracs = []

    logger.info("Rendering %d novel views...", len(cameras))
    t0 = time.time()
    for i, cam in enumerate(cameras):
        rgb, mask = render_with_hole_mask(gaussians, cam, alpha_threshold=config.alpha_threshold)
        renders.append(rgb)
        masks.append(mask)
        frac = float(mask.mean())
        hole_fracs.append(frac)

        # Save every 3rd view (one per depth tier per direction)
        if i % 3 == 0:
            idx = i // 3
            save_rgb(rgb, outdir / f"novel_view_{idx:02d}.png")
            save_mask(mask, outdir / f"hole_mask_{idx:02d}.png")

            # Overlay: holes in red on render
            overlay = rgb.copy()
            overlay[mask > 0.5] = [1.0, 0.0, 0.0]
            save_rgb(overlay, outdir / f"holes_overlay_{idx:02d}.png")

    render_time = time.time() - t0
    logger.info("Rendered %d views in %.1fs", len(cameras), render_time)

    # --- Phase 1c: Camera selection analysis ---
    avg_hole = float(np.mean(hole_fracs))
    max_hole = float(np.max(hole_fracs))
    repair_indices = select_repair_cameras(
        cameras, masks,
        min_hole_fraction=config.min_hole_fraction,
        max_cameras=config.max_repair_cameras,
    )
    logger.info("Avg hole fraction: %.4f, Max: %.4f", avg_hole, max_hole)
    logger.info("Selected %d repair cameras (of %d)", len(repair_indices), len(cameras))

    # --- Phase 1d: Cubemap extraction ---
    logger.info("Extracting cubemap views...")
    cubemap_faces, cubemap_cameras = extract_cubemap_views(
        panorama, depth_map, face_size=args.resolution,
    )
    face_names = ["front", "back", "right", "left", "top", "bottom"]
    for i, (face, name) in enumerate(zip(cubemap_faces, face_names)):
        save_rgb(face, outdir / f"cubemap_{name}.png")

    # --- Render cubemap cameras through the Gaussians to compare ---
    logger.info("Rendering cubemap views through Gaussians for comparison...")
    for i, (cam, face, name) in enumerate(zip(cubemap_cameras, cubemap_faces, face_names)):
        rgb, mask = render_with_hole_mask(gaussians, cam, alpha_threshold=config.alpha_threshold)
        save_side_by_side(face, rgb, outdir / f"compare_cubemap_{name}.png",
                          labels=[f"Panorama ({name})", f"GS render ({name})"])
        save_mask(mask, outdir / f"cubemap_holes_{name}.png")
        hole_pct = float(mask.mean()) * 100
        logger.info("  %s: hole=%.1f%%", name, hole_pct)

    # --- Camera R/T sanity check ---
    logger.info("\n--- Camera Convention Check ---")
    test_cam = CameraPose(
        position=np.array([0.0, 0.0, 0.0]),
        look_at=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=90.0, width=512, height=512,
    )
    R, T = _camera_to_RT(test_cam)
    logger.info("Identity camera R:\n%s", R)
    logger.info("Identity camera T: %s", T)
    logger.info("Expected R ≈ I, T ≈ [0,0,0]")

    # Check a translated camera
    test_cam2 = CameraPose(
        position=np.array([1.0, 0.0, 0.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0, width=512, height=512,
    )
    R2, T2 = _camera_to_RT(test_cam2)
    # camera_center should reconstruct to position
    W2C = np.eye(4)
    W2C[:3, :3] = R2
    W2C[:3, 3] = T2
    C2W = np.linalg.inv(W2C)
    reconstructed_pos = C2W[:3, 3]
    logger.info("Camera at [1,0,0] -> reconstructed position: %s (should be [1,0,0])",
                reconstructed_pos)
    pos_error = np.linalg.norm(reconstructed_pos - test_cam2.position)
    logger.info("Position reconstruction error: %.6f", pos_error)

    # --- Write summary ---
    summary_path = outdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Diagnostic Render Summary\n")
        f.write(f"========================\n\n")
        f.write(f"PLY: {args.ply}\n")
        f.write(f"Gaussians: {n_gaussians:,}\n")
        f.write(f"Panorama: {panorama.shape}\n")
        f.write(f"Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]\n\n")
        f.write(f"Novel Views: {len(cameras)}\n")
        f.write(f"Resolution: {args.resolution}\n")
        f.write(f"Avg hole fraction: {avg_hole:.4f} ({avg_hole*100:.1f}%)\n")
        f.write(f"Max hole fraction: {max_hole:.4f} ({max_hole*100:.1f}%)\n")
        f.write(f"Repair cameras selected: {len(repair_indices)}\n\n")
        f.write(f"Current config parameters:\n")
        f.write(f"  densify_grad_threshold: {config.densify_grad_threshold}\n")
        f.write(f"  alpha_threshold: {config.alpha_threshold}\n")
        f.write(f"  convergence_threshold: {config.convergence_threshold}\n")
        f.write(f"  min_hole_fraction: {config.min_hole_fraction}\n\n")
        f.write(f"Per-view hole fractions:\n")
        for i, frac in enumerate(hole_fracs):
            marker = " <-- repair" if i in repair_indices else ""
            f.write(f"  cam {i:3d}: {frac:.4f} ({frac*100:.1f}%){marker}\n")

    logger.info("\nDiagnostics saved to %s", outdir)
    logger.info("Summary: %s", summary_path)

    # Cleanup GPU
    del gaussians
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
