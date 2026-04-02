"""Phase 3: 3DGS distillation — optimize Gaussians against repaired images."""

import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# Add GSFix3D to path
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


def distill_to_gaussians(
    gaussians,
    repaired_images,
    cameras,
    hole_masks,
    original_images=None,
    original_cameras=None,
    num_iterations=3000,
    densify_interval=100,
    densify_grad_threshold=0.0002,
    lr_position=0.00016,
    lr_feature=0.0025,
    lr_opacity=0.05,
    lr_scaling=0.005,
    lr_rotation=0.001,
    original_view_ratio=0.3,
    prune_opacity_threshold=0.005,
):
    """Optimize 3DGS to match repaired images via differentiable rendering.

    Follows GSFix3D's refine_gs.py pattern:
    1. For each repaired view: render, L1+SSIM loss, backprop, densify/prune
    2. Then optimize over mixed dataset (repaired + original views)

    Depth alignment is emergent — the optimizer places Gaussians at whatever
    3D positions produce correct 2D appearance across multiple viewpoints.
    """
    if gaussians is None:
        logger.warning("distill_to_gaussians: gaussians is None, skipping")
        return gaussians

    from gs.camera import Camera as GSCamera
    from gs.gaussian_renderer import render as gs_render
    from gs.loss_utils import l1_loss, ssim
    from gs.arguments import OptimizationParams

    from .camera_rig import _camera_to_RT

    # Create a mock parser for OptimizationParams
    import argparse
    mock_parser = argparse.ArgumentParser()
    optim_params = OptimizationParams(mock_parser)
    # Override with our settings
    optim_params.position_lr_init = lr_position
    optim_params.position_lr_final = lr_position * 0.01
    optim_params.position_lr_max_steps = num_iterations
    optim_params.feature_lr = lr_feature
    optim_params.opacity_lr = lr_opacity
    optim_params.scaling_lr = lr_scaling
    optim_params.rotation_lr = lr_rotation
    optim_params.densify_grad_threshold = densify_grad_threshold
    optim_params.prune_opacity_threshold = prune_opacity_threshold

    # Setup optimizer
    gaussians.training_setup(optim_params)

    # Pipe mock for renderer
    class PipeMock:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    bg = torch.zeros(3, device="cuda")
    to_tensor = transforms.ToTensor()

    def _camera_to_gs(cam):
        """Convert CameraPose to GSFix3D Camera."""
        R, T = _camera_to_RT(cam)
        fov_rad = math.radians(cam.fov_deg)
        aspect = cam.width / cam.height
        fov_x = 2 * math.atan(math.tan(fov_rad / 2) * aspect)
        return GSCamera(R=R, T=T, FoVx=fov_x, FoVy=fov_rad,
                        width=cam.width, height=cam.height)

    # Convert images to tensors (3, H, W) on CUDA
    repair_tensors = [
        torch.from_numpy(img).permute(2, 0, 1).cuda() for img in repaired_images
    ]
    repair_gs_cams = [_camera_to_gs(cam) for cam in cameras]

    orig_tensors = None
    orig_gs_cams = None
    if original_images:
        orig_tensors = [
            torch.from_numpy(img).permute(2, 0, 1).cuda() for img in original_images
        ]
        orig_gs_cams = [_camera_to_gs(cam) for cam in original_cameras]

    n_repair = len(repair_tensors)
    n_orig = len(orig_tensors) if orig_tensors else 0
    total_views = n_repair + n_orig

    logger.info(f"Distillation: {num_iterations} iters, "
                f"{n_repair} repaired + {n_orig} original views")

    # --- Phase A: Per-view optimization on repaired images ---
    # (Following refine_gs.py: iterate over each fixed image with multiple steps)
    iters_per_view = min(20, num_iterations // max(n_repair, 1))

    for view_idx in range(n_repair):
        gt_image = repair_tensors[view_idx]
        gs_cam = repair_gs_cams[view_idx]

        for step in range(iters_per_view):
            render_pkg = gs_render(gs_cam, gaussians, PipeMock(), bg)
            rendered = render_pkg["render"]
            viewspace_points = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]

            loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * (1 - ssim(rendered, gt_image))
            loss.backward()

            gaussians.add_densification_stats(viewspace_points, visibility_filter)

            if step % 5 == 0:
                gaussians.densify(densify_grad_threshold)

            if step == iters_per_view - 1:
                gaussians.prune(prune_opacity_threshold)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if view_idx % 5 == 0:
            logger.info(f"  Phase A: view {view_idx+1}/{n_repair}, "
                        f"loss={loss.item():.4f}")

    # --- Phase B: Mixed optimization over all views ---
    remaining_iters = max(50, num_iterations - iters_per_view * n_repair)
    all_tensors = list(repair_tensors) + (list(orig_tensors) if orig_tensors else [])
    all_cams = list(repair_gs_cams) + (list(orig_gs_cams) if orig_gs_cams else [])

    for step in range(remaining_iters):
        idx = random.randint(0, len(all_tensors) - 1)
        gt_image = all_tensors[idx]
        gs_cam = all_cams[idx]

        rendered = gs_render(gs_cam, gaussians, PipeMock(), bg)["render"]
        loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * (1 - ssim(rendered, gt_image))
        loss.backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            logger.info(f"  Phase B: step {step}/{remaining_iters}, "
                        f"loss={loss.item():.4f}")

    final_count = gaussians.get_xyz.shape[0]
    logger.info(f"Distillation complete. Gaussians: {final_count}")
    return gaussians
