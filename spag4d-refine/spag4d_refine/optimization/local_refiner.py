"""Masked gsplat optimization for Gaussian refinement.

Uses differentiable gsplat rendering to optimize Gaussian parameters
against Klein-synthesized target images. Original Gaussians are
anchored (low learning rate + anchor loss) while seeded Gaussians
are free to move, recolor, and resize to match the targets.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from ..camera.pinhole import PinholeCamera
from ..gaussian.cloud import GaussianCloud, SH_C0
from ..gaussian.provenance import GaussianSource

logger = logging.getLogger(__name__)


def _opengl_w2c_to_opencv(w2c_gl: np.ndarray) -> np.ndarray:
    """Convert OpenGL w2c (Y-up, -Z forward) to OpenCV (Y-down, +Z forward)."""
    w2c_cv = w2c_gl.copy()
    w2c_cv[1, :] = -w2c_gl[1, :]  # flip Y
    w2c_cv[2, :] = -w2c_gl[2, :]  # flip Z
    return w2c_cv


def refine_gaussians(
    cloud: GaussianCloud,
    target_images: List[np.ndarray],
    cameras: List[PinholeCamera],
    iterations: int = 4000,
    anchor_loss_weight: float = 8.0,
    original_lr_scale: float = 0.05,
    device: str = "cuda",
) -> GaussianCloud:
    """
    Constrained optimization of Gaussian parameters using gsplat.

    Freezes (or heavily penalizes) original Gaussians while optimizing
    seeded/promoted ones to match target images.

    Args:
        cloud: GaussianCloud with mixed provenance
        target_images: List of [H, W, 3] float32 sRGB target images
        cameras: Corresponding cameras for each target image
        iterations: Number of optimization iterations
        anchor_loss_weight: Weight for anchor loss (prevents original drift)
        original_lr_scale: Learning rate multiplier for original Gaussians
        device: Torch device

    Returns:
        Optimized GaussianCloud
    """
    from gsplat import rasterization

    if len(target_images) == 0:
        logger.warning("No target images for optimization, returning cloud unchanged")
        return cloud

    dev = torch.device(device)

    # Prepare parameters from cloud (gsplat format: WXYZ quats, SH0 colors, linear scales)
    gsplat_data = cloud.to_gsplat()
    means = gsplat_data["means"].to(dev).requires_grad_(True)
    scales_raw = torch.log(gsplat_data["scales"].to(dev).clamp(min=1e-7))
    scales_raw.requires_grad_(True)
    quats = gsplat_data["quats"].to(dev).requires_grad_(True)

    # Colors are SH0 coefficients from to_gsplat(). Optimize in SH0 space.
    sh0_colors = gsplat_data["colors"].to(dev).requires_grad_(True)

    opacities_raw = gsplat_data["opacities"].to(dev).clamp(1e-6, 1 - 1e-6)
    opacities_logit = torch.log(opacities_raw / (1 - opacities_raw))
    opacities_logit.requires_grad_(True)

    # Store anchors for original Gaussians
    is_original = torch.from_numpy(
        cloud.provenance == GaussianSource.ORIGINAL
    ).bool().to(dev)
    anchor_means = means.detach().clone()
    anchor_colors = sh0_colors.detach().clone()

    # Per-parameter learning rates
    base_lr = 1e-3
    params = [
        {"params": [means], "lr": base_lr * 0.1},
        {"params": [scales_raw], "lr": base_lr * 0.05},
        {"params": [quats], "lr": base_lr * 0.1},
        {"params": [sh0_colors], "lr": base_lr},
        {"params": [opacities_logit], "lr": base_lr * 0.5},
    ]
    optimizer = torch.optim.Adam(params)

    # Target images stay in sRGB [0,1] — gsplat with sh_degree=0 outputs
    # SH_C0 * sh_dc + 0.5 which equals sRGB. Loss is computed in sRGB space.
    targets = [
        torch.from_numpy(img).float().to(dev) for img in target_images
    ]

    # Camera matrices: convert OpenGL → OpenCV for gsplat
    K_list = [
        torch.from_numpy(cam.K).float().unsqueeze(0).to(dev) for cam in cameras
    ]
    viewmat_list = [
        torch.from_numpy(_opengl_w2c_to_opencv(cam.w2c)).float().unsqueeze(0).to(dev)
        for cam in cameras
    ]

    n_views = len(targets)
    log_interval = max(iterations // 8, 100)

    logger.info(
        f"Optimization: {iterations} iters, {n_views} views, "
        f"{int(is_original.sum())} original + "
        f"{len(cloud) - int(is_original.sum())} seeded Gaussians"
    )

    for step in range(iterations):
        optimizer.zero_grad()

        # Cycle through views
        view_idx = step % n_views
        target = targets[view_idx]
        H, W = target.shape[:2]
        K = K_list[view_idx]
        viewmat = viewmat_list[view_idx]

        # Decode parameters
        scales = torch.exp(scales_raw)
        opacities = torch.sigmoid(opacities_logit)

        # Render (gsplat outputs SH0-decoded values)
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=scales,
            opacities=opacities,
            colors=sh0_colors.unsqueeze(1),
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=0,
        )

        rendered = renders[0, :, :, :3]

        # L1 photometric loss (in SH0 space)
        photo_loss = (rendered - target).abs().mean()

        # Anchor loss: penalize drift of original Gaussians
        means_diff = (means[is_original] - anchor_means[is_original]).pow(2).sum(dim=-1).mean()
        colors_diff = (sh0_colors[is_original] - anchor_colors[is_original]).pow(2).sum(dim=-1).mean()
        anchor_loss = anchor_loss_weight * (means_diff + colors_diff)

        loss = photo_loss + anchor_loss

        loss.backward()

        # Scale down gradients for original Gaussians
        if means.grad is not None:
            means.grad[is_original] *= original_lr_scale
        if sh0_colors.grad is not None:
            sh0_colors.grad[is_original] *= original_lr_scale
        if scales_raw.grad is not None:
            scales_raw.grad[is_original] *= original_lr_scale

        optimizer.step()

        if step % log_interval == 0:
            logger.info(
                f"  Step {step}/{iterations}: "
                f"photo={photo_loss.item():.4f}, anchor={anchor_loss.item():.4f}"
            )

    # Build output cloud: convert from gsplat space back to internal conventions
    with torch.no_grad():
        final_means = means.cpu().numpy()
        final_scales = torch.exp(scales_raw).cpu().numpy()
        final_quats_wxyz = (quats / quats.norm(dim=-1, keepdim=True)).cpu().numpy()
        final_opacities = torch.sigmoid(opacities_logit).cpu().numpy()

        # SH0 → sRGB for internal storage
        final_colors_srgb = (sh0_colors * SH_C0 + 0.5).clamp(0, 1).cpu().numpy()

        # WXYZ → XYZW for internal convention
        final_quats = np.stack([
            final_quats_wxyz[:, 1],
            final_quats_wxyz[:, 2],
            final_quats_wxyz[:, 3],
            final_quats_wxyz[:, 0],
        ], axis=-1)

    logger.info(
        f"Optimization complete. Final photo_loss={photo_loss.item():.4f}"
    )

    return GaussianCloud(
        means=final_means,
        scales=final_scales,
        quats=final_quats,
        colors=final_colors_srgb,
        opacities=final_opacities,
        provenance=cloud.provenance.copy(),
    )
