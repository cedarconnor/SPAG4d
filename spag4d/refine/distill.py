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
    densify_grad_threshold=0.005,
    lr_position=0.00032,
    lr_feature=0.0025,
    lr_opacity=0.025,
    lr_scaling=0.002,
    lr_rotation=0.001,
    original_view_ratio=0.3,
    prune_opacity_threshold=0.005,
    iters_per_view=20,
    kf_iters=50,
    protect_original_count=0,
    tier2_weight=0.20,
):
    """Optimize 3DGS to match repaired images via differentiable rendering.

    Two modes controlled by ``protect_original_count``:

    **GSFix3D mode** (protect_original_count=0, default):
        GSFixer produces full-frame inpainted images so the loss covers the
        entire image.  Original Gaussians get 0.1x gradient scaling.
        Densification is enabled during Phase A.

    **OmniRoam / hole-fill mode** (protect_original_count > 0):
        Only fills holes — original Gaussians are completely frozen in Phase A
        and the loss is masked to hole regions.  Phase B uses only tier-1
        (source cubemap) views to anchor consistency.  No densification.
    """
    if gaussians is None:
        logger.warning("distill_to_gaussians: gaussians is None, skipping")
        return gaussians

    from gs.camera import Camera as GSCamera
    from gs.gaussian_renderer import render as gs_render
    from gs.loss_utils import l1_loss, ssim
    from gs.arguments import OptimizationParams

    from .camera_rig import _camera_to_RT

    hole_fill_mode = protect_original_count > 0
    total_count_before = gaussians.get_xyz.shape[0]

    # Create a mock parser for OptimizationParams
    import argparse
    mock_parser = argparse.ArgumentParser()
    optim_params = OptimizationParams(mock_parser)
    optim_params.position_lr_init = lr_position
    optim_params.position_lr_final = lr_position * 0.01
    optim_params.position_lr_max_steps = iters_per_view * len(repaired_images) + kf_iters
    optim_params.feature_lr = lr_feature
    optim_params.opacity_lr = lr_opacity
    optim_params.scaling_lr = lr_scaling
    optim_params.rotation_lr = lr_rotation
    optim_params.densify_grad_threshold = densify_grad_threshold
    optim_params.prune_opacity_threshold = prune_opacity_threshold

    gaussians.training_setup(optim_params)

    if not hole_fill_mode:
        # GSFix3D: gentle 0.1x scaling for originals
        _apply_original_lr_scaling(gaussians, total_count_before, scale=0.1)

    # Pipe mock for renderer
    class PipeMock:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    bg = torch.zeros(3, device="cuda")

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

    # Convert hole masks to tensors
    hole_mask_tensors = []
    for i, mask in enumerate(hole_masks):
        if mask is not None:
            hole_mask_tensors.append(torch.from_numpy(mask).cuda())
        else:
            hole_mask_tensors.append(None)

    orig_tensors = None
    orig_gs_cams = None
    if original_images:
        orig_tensors = [
            torch.from_numpy(img).permute(2, 0, 1).cuda() for img in original_images
        ]
        orig_gs_cams = [_camera_to_gs(cam) for cam in original_cameras]

    n_repair = len(repair_tensors)
    n_orig = len(orig_tensors) if orig_tensors else 0

    mode_label = "hole-fill" if hole_fill_mode else "GSFix3D"
    logger.info(f"Distillation ({mode_label}): {iters_per_view} iters/view x "
                f"{n_repair} repaired + {kf_iters} anchor iters ({n_orig} original views)")

    # ── Phase A: Per-view optimization on repaired images ──
    for view_idx in range(n_repair):
        gt_image = repair_tensors[view_idx]
        gs_cam = repair_gs_cams[view_idx]
        mask_t = hole_mask_tensors[view_idx] if view_idx < len(hole_mask_tensors) else None

        for step in range(iters_per_view):
            render_pkg = gs_render(gs_cam, gaussians, PipeMock(), bg)
            rendered = render_pkg["render"]

            if hole_fill_mode and mask_t is not None:
                # Loss ONLY in hole regions, scaled by configured tier-2 weight
                loss = compute_weighted_loss(rendered, gt_image,
                                            tier2_weight=tier2_weight, hole_mask=mask_t)
            else:
                loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * (1 - ssim(rendered, gt_image))

            loss.backward()

            if hole_fill_mode:
                # Completely freeze original Gaussians — zero their gradients
                _zero_original_grads(gaussians, protect_original_count)
            else:
                # GSFix3D mode: densify (not used in hole-fill mode)
                viewspace_points = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
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

    # ── Phase B: Anchor optimization ──
    if hole_fill_mode:
        # Hole-fill mode: ONLY tier-1 (source cubemap) views — no tier-2
        # This anchors consistency without dragging originals toward OmniRoam
        anchor_tensors = orig_tensors or []
        anchor_cams = orig_gs_cams or []
    else:
        # GSFix3D mode: mixed (repaired + original) views
        anchor_tensors = list(repair_tensors) + (list(orig_tensors) if orig_tensors else [])
        anchor_cams = list(repair_gs_cams) + (list(orig_gs_cams) if orig_gs_cams else [])

    if anchor_tensors:
        for step in range(kf_iters):
            indices = list(range(len(anchor_tensors)))
            random.shuffle(indices)

            for idx in indices:
                gt_image = anchor_tensors[idx]
                gs_cam = anchor_cams[idx]

                rendered = gs_render(gs_cam, gaussians, PipeMock(), bg)["render"]
                loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * (1 - ssim(rendered, gt_image))
                loss.backward()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                logger.info(f"  Phase B: pass {step}/{kf_iters}, loss={loss.item():.4f}")

    final_count = gaussians.get_xyz.shape[0]
    delta = final_count - total_count_before
    logger.info(f"Distillation complete. Gaussians: {total_count_before} -> {final_count} "
                f"({'+'if delta >= 0 else ''}{delta})")
    return gaussians


def compute_weighted_loss(
    rendered: "torch.Tensor",
    gt: "torch.Tensor",
    tier2_weight: float = 1.0,
    hole_mask: "torch.Tensor" = None,
) -> "torch.Tensor":
    """Compute L1 loss with tier-2 weighting and optional hole masking.

    Args:
        rendered: (3, H, W) rendered image tensor.
        gt: (3, H, W) ground truth image tensor.
        tier2_weight: Weight multiplier (1.0 for tier-1, 0.15-0.30 for tier-2).
        hole_mask: Optional (H, W) float tensor. If provided, loss is only
            computed in regions where mask > 0.5 (hole regions).

    Returns:
        Scalar loss tensor.
    """
    import torch

    if hole_mask is not None:
        mask = (hole_mask > 0.5).unsqueeze(0).float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        rendered_m = rendered * mask
        gt_m = gt * mask
        n_pixels = mask.sum() * 3
        l1 = torch.abs(rendered_m - gt_m).sum() / n_pixels
        loss = l1
    else:
        l1 = torch.abs(rendered - gt).mean()
        loss = l1

    return loss * tier2_weight


def _zero_original_grads(gaussians, original_count):
    """Zero out gradients for original Gaussians (indices < original_count).

    Called after backward() in hole-fill mode to completely freeze the
    original Gaussians while allowing gap-seed and new Gaussians to learn.
    """
    for param_group in gaussians.optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None and len(param.shape) >= 1 and param.shape[0] >= original_count:
                param.grad[:original_count] = 0.0


def _apply_original_lr_scaling(gaussians, initial_count, scale=0.1):
    """Reduce LR for original Gaussians by scaling their gradients.

    Used in GSFix3D mode where originals should update gently (not frozen).
    """
    if not hasattr(gaussians, 'optimizer') or gaussians.optimizer is None:
        return

    def _scale_grad_hook(grad, count=initial_count, s=scale):
        if grad is None:
            return grad
        scaled = grad.clone()
        if len(scaled.shape) >= 1 and scaled.shape[0] > count:
            scaled[:count] *= s
        return scaled

    for param_group in gaussians.optimizer.param_groups:
        for param in param_group['params']:
            if param.requires_grad and param.shape[0] >= initial_count:
                param.register_hook(_scale_grad_hook)

    logger.info(f"Applied {scale}x gradient scaling to {initial_count} original Gaussians")
