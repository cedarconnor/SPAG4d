"""Top-level refinement pipeline orchestrator."""
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import RefineConfig
from .camera_rig import (
    generate_camera_rig, render_with_hole_mask,
    select_repair_cameras, extract_cubemap_views,
)
from .mesh_extract import extract_conditioning_mesh
from .gsfixer_adapter import GSFixerAdapter
from .distill import distill_to_gaussians
from .provenance import tag_gaussian_provenance
from .format_compat import load_gaussians_from_ply, save_gaussians_to_ply

logger = logging.getLogger(__name__)


def refine_splat(
    ply_path: str,
    panorama_path: str,
    depth_map: np.ndarray,
    max_iterations: int = 3,
    num_cameras: int = 36,
    finetune_steps: int = 500,
    output_path: str = None,
    config: RefineConfig = None,
    progress_callback: Optional[Callable] = None,
    diagnostics_dir: Optional[str] = None,
) -> dict:
    """Full GSFix3D refinement pipeline."""
    config = config or RefineConfig()
    config.max_iterations = max_iterations
    config.finetune_steps = finetune_steps
    start_time = time.time()

    def report(stage, pct, iteration=0):
        logger.info(f"[refine] iter={iteration} stage={stage} pct={pct}")
        if progress_callback:
            progress_callback(iteration, stage, pct)

    diag_dir = Path(diagnostics_dir) if diagnostics_dir else None
    if diag_dir:
        diag_dir.mkdir(parents=True, exist_ok=True)

    # Load panorama
    from PIL import Image
    panorama = np.array(Image.open(panorama_path)).astype(np.float32) / 255.0
    if panorama.ndim == 3 and panorama.shape[2] == 4:
        panorama = panorama[:, :, :3]

    report("camera_rig", 5)

    # Phase 1: Camera rig
    num_dirs = max(num_cameras // 3, 4)
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth_map,
        num_directions=num_dirs,
        num_depths=3,
        fov_deg=config.camera_fov,
        translation_fracs=config.translation_fracs,
        resolution=config.render_resolution,
    )

    report("mesh_extract", 10)
    mesh = extract_conditioning_mesh(
        depth_map=depth_map, panorama=panorama,
        simplify_ratio=config.mesh_simplify_ratio,
    )

    cubemap_faces, cubemap_cameras = extract_cubemap_views(
        panorama, depth_map, face_size=config.render_resolution,
    )

    report("finetune", 15)

    # Load Gaussians early so we can render cubemap views for fine-tuning
    gaussians = load_gaussians_from_ply(ply_path, device="cuda")

    # Render GS from cubemap cameras — these have holes, unlike the panorama faces.
    # Fine-tuning needs (holey GS render → clean panorama face) pairs so the model
    # learns to fill holes, not just identity mapping.
    cubemap_gs_renders = []
    for cam in cubemap_cameras:
        rgb, _ = render_with_hole_mask(gaussians, cam, alpha_threshold=config.alpha_threshold)
        cubemap_gs_renders.append(rgb)

    # Phase 2: GSFixer
    gsfixer = GSFixerAdapter(checkpoint_path=config.gsfixer_checkpoint, device="cuda")
    gsfixer.load()
    gsfixer.finetune(
        gs_renders=cubemap_gs_renders, gt_images=cubemap_faces,
        mesh=mesh, cameras=cubemap_cameras,
        train_steps=config.finetune_steps, learning_rate=config.finetune_lr,
    )

    report("render_holes", 30)

    # Track initial count for provenance
    initial_gaussian_count = 0
    if gaussians is not None and hasattr(gaussians, 'get_xyz'):
        initial_gaussian_count = gaussians.get_xyz.shape[0]

    initial_hole_frac = None
    avg_hole_frac = 0.0
    iteration = 0

    for iteration in range(config.max_iterations):
        iter_num = iteration + 1
        report("render_holes", 30 + iteration * 20, iter_num)

        renders, masks = [], []
        for cam in cameras:
            rgb, mask = render_with_hole_mask(gaussians, cam, alpha_threshold=config.alpha_threshold)
            renders.append(rgb)
            masks.append(mask)

        avg_hole_frac = float(np.mean([m.mean() for m in masks]))
        if initial_hole_frac is None:
            initial_hole_frac = avg_hole_frac

        if avg_hole_frac < config.convergence_threshold:
            break

        repair_indices = select_repair_cameras(
            cameras, masks,
            min_hole_fraction=config.min_hole_fraction,
            max_cameras=config.max_repair_cameras,
        )

        if len(repair_indices) == 0:
            break

        report("gsfixer_inference", 50 + iteration * 15, iter_num)

        repair_renders = [renders[i] for i in repair_indices]
        repair_masks = [masks[i] for i in repair_indices]
        repair_cams = [cameras[i] for i in repair_indices]

        repaired = gsfixer.infer(
            gs_renders=repair_renders, hole_masks=repair_masks,
            mesh=mesh, cameras=repair_cams,
            num_steps=config.inference_steps, guidance_scale=config.guidance_scale,
        )

        if diag_dir:
            _save_diagnostics(diag_dir, iter_num, repair_renders, repair_masks, repaired, repair_indices)

        report("distill", 70 + iteration * 10, iter_num)

        gaussians = distill_to_gaussians(
            gaussians=gaussians, repaired_images=repaired,
            cameras=repair_cams, hole_masks=repair_masks,
            original_images=cubemap_faces, original_cameras=cubemap_cameras,
            densify_grad_threshold=config.densify_grad_threshold,
            iters_per_view=20,   # Match refine_gs.py
            kf_iters=50,         # Match refine_gs.py (conservative)
        )

        tag_gaussian_provenance(gaussians, initial_gaussian_count)

    gsfixer.unload()

    # Export refined PLY
    output_path = output_path or ply_path.replace('.ply', '_refined.ply')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_gaussians_to_ply(gaussians, output_path)

    report("distill", 100, config.max_iterations)

    gaussian_count = 0
    if gaussians is not None and hasattr(gaussians, 'get_xyz'):
        gaussian_count = gaussians.get_xyz.shape[0]

    total_time = time.time() - start_time
    logger.info(f"[refine] Done in {total_time:.1f}s. "
                f"Holes: {initial_hole_frac:.4f} -> {avg_hole_frac:.4f}, "
                f"Gaussians: {gaussian_count}")

    return {
        "refined_ply_path": output_path,
        "initial_hole_fraction": initial_hole_frac or 0.0,
        "final_hole_fraction": avg_hole_frac,
        "gaussians_count": gaussian_count,
        "iterations_used": min(iteration + 1, config.max_iterations),
        "total_time": round(total_time, 1),
    }


def _save_diagnostics(diag_dir, iteration, renders, masks, repaired, indices):
    """Save diagnostic images for UI gallery."""
    from PIL import Image

    for i, (render, mask, repair, cam_idx) in enumerate(
        zip(renders, masks, repaired, indices)
    ):
        prefix = f"r{iteration}_cam{cam_idx}"
        img = Image.fromarray((render * 255).clip(0, 255).astype(np.uint8))
        img.save(diag_dir / f"{prefix}_splat.png")
        mask_img = Image.fromarray((mask * 255).clip(0, 255).astype(np.uint8))
        mask_img.save(diag_dir / f"{prefix}_mask.png")
        rep_img = Image.fromarray((repair * 255).clip(0, 255).astype(np.uint8))
        rep_img.save(diag_dir / f"{prefix}_repaired.png")
