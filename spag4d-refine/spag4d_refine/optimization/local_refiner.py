"""Masked gsplat optimization for Gaussian refinement."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from ..camera.pinhole import PinholeCamera
from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource

logger = logging.getLogger(__name__)


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
        target_images: List of [H, W, 3] float32 target RGB images
        cameras: Corresponding cameras for each target image
        iterations: Number of optimization iterations
        anchor_loss_weight: Weight for anchor loss (prevents original drift)
        original_lr_scale: Learning rate multiplier for original Gaussians
        device: Torch device

    Returns:
        Optimized GaussianCloud
    """
    from gsplat import rasterization

    dev = torch.device(device)

    # Prepare parameters
    gsplat_data = cloud.to_gsplat()
    means = gsplat_data["means"].to(dev).requires_grad_(True)
    scales_raw = torch.log(gsplat_data["scales"].to(dev).clamp(min=1e-7))
    scales_raw.requires_grad_(True)
    quats = gsplat_data["quats"].to(dev).requires_grad_(True)
    colors = gsplat_data["colors"].to(dev).requires_grad_(True)

    opacities_raw = gsplat_data["opacities"].to(dev).clamp(1e-6, 1 - 1e-6)
    opacities_logit = torch.log(opacities_raw / (1 - opacities_raw))
    opacities_logit.requires_grad_(True)

    # Store anchors for original Gaussians
    is_original = torch.from_numpy(
        cloud.provenance == GaussianSource.ORIGINAL
    ).bool().to(dev)
    anchor_means = means.detach().clone()
    anchor_colors = colors.detach().clone()

    # Per-parameter learning rates
    base_lr = 1e-3
    params = [
        {"params": [means], "lr": base_lr * 0.1},
        {"params": [scales_raw], "lr": base_lr * 0.05},
        {"params": [quats], "lr": base_lr * 0.1},
        {"params": [colors], "lr": base_lr},
        {"params": [opacities_logit], "lr": base_lr * 0.5},
    ]
    optimizer = torch.optim.Adam(params)

    # Prepare target images as tensors
    targets = [
        torch.from_numpy(img).float().to(dev) for img in target_images
    ]
    K_list = [
        torch.from_numpy(cam.K).float().unsqueeze(0).to(dev) for cam in cameras
    ]
    viewmat_list = [
        torch.from_numpy(cam.w2c).float().unsqueeze(0).to(dev) for cam in cameras
    ]

    n_views = len(targets)

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

        # Render
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=scales,
            opacities=opacities,
            colors=colors.unsqueeze(1),
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=0,
        )

        rendered_rgb = renders[0, :, :, :3]

        # L1 photometric loss
        photo_loss = (rendered_rgb - target).abs().mean()

        # Anchor loss: penalize drift of original Gaussians
        means_diff = (means[is_original] - anchor_means[is_original]).pow(2).sum(dim=-1).mean()
        colors_diff = (colors[is_original] - anchor_colors[is_original]).pow(2).sum(dim=-1).mean()
        anchor_loss = anchor_loss_weight * (means_diff + colors_diff)

        loss = photo_loss + anchor_loss

        loss.backward()

        # Scale down gradients for original Gaussians
        if means.grad is not None:
            means.grad[is_original] *= original_lr_scale
        if colors.grad is not None:
            colors.grad[is_original] *= original_lr_scale
        if scales_raw.grad is not None:
            scales_raw.grad[is_original] *= original_lr_scale

        optimizer.step()

        if step % 500 == 0:
            logger.info(
                f"  Step {step}/{iterations}: "
                f"photo={photo_loss.item():.4f}, anchor={anchor_loss.item():.4f}"
            )

    # Build output cloud
    with torch.no_grad():
        final_means = means.cpu().numpy()
        final_scales = torch.exp(scales_raw).cpu().numpy()
        final_quats_wxyz = (quats / quats.norm(dim=-1, keepdim=True)).cpu().numpy()
        final_colors = colors.clamp(0, 1).cpu().numpy()
        final_opacities = torch.sigmoid(opacities_logit).cpu().numpy()

        # WXYZ → XYZW for internal convention
        final_quats = np.stack([
            final_quats_wxyz[:, 1],
            final_quats_wxyz[:, 2],
            final_quats_wxyz[:, 3],
            final_quats_wxyz[:, 0],
        ], axis=-1)

    return GaussianCloud(
        means=final_means,
        scales=final_scales,
        quats=final_quats,
        colors=final_colors,
        opacities=final_opacities,
        provenance=cloud.provenance.copy(),
    )
