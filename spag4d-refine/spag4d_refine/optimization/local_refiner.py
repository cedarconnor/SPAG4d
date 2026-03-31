"""Masked gsplat optimization for Gaussian refinement.

Uses differentiable gsplat rendering to optimize Gaussian parameters
against Klein-synthesized target images. Original Gaussians are
anchored (low learning rate + anchor loss) while seeded Gaussians
are free to move, recolor, and resize to match the targets.

Loss components:
  - L1 photometric loss (masked to gap regions)
  - DINOv2 semantic feature loss (perceptual similarity)
  - Pearson depth correlation loss (scale-invariant geometry)
  - Anchor loss (prevents original Gaussian drift)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..camera.pinhole import PinholeCamera
from ..gaussian.cloud import GaussianCloud, SH_C0
from ..gaussian.provenance import GaussianSource

logger = logging.getLogger(__name__)

# Lazy-loaded DINOv2 model (shared across calls)
_dino_model = None


def _get_dino_model(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-S/14 once and cache it."""
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True, verbose=False,
        )
        _dino_model.eval()
        logger.info("Loaded DINOv2 ViT-S/14 for semantic feature loss")
    _dino_model = _dino_model.to(device)
    return _dino_model


def _extract_dino_features(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Extract DINOv2 patch-level features from an image.

    Args:
        model: DINOv2 model
        image: [H, W, 3] float tensor in [0, 1]
        patch_size: DINOv2 patch size (14 for ViT-S/14)

    Returns:
        [H_p, W_p, D] feature map (D=384 for ViT-S)
    """
    H, W = image.shape[:2]
    # Crop to multiple of patch_size
    H_p = (H // patch_size) * patch_size
    W_p = (W // patch_size) * patch_size
    img_crop = image[:H_p, :W_p]

    # DINOv2 expects [B, 3, H, W] normalized with ImageNet stats
    img_t = img_crop.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    img_norm = (img_t - mean) / std

    # Extract patch tokens (skip CLS token)
    features = model.forward_features(img_norm)
    patch_tokens = features["x_norm_patchtokens"]  # [1, N_patches, D]

    n_h = H_p // patch_size
    n_w = W_p // patch_size
    feat_map = patch_tokens.reshape(1, n_h, n_w, -1).squeeze(0)  # [H_p, W_p, D]
    return feat_map


def _pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient between two 1D tensors."""
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    num = (x_centered * y_centered).sum()
    denom = (x_centered.pow(2).sum() * y_centered.pow(2).sum()).sqrt().clamp(min=1e-8)
    return num / denom


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
    gap_masks: Optional[List[np.ndarray]] = None,
    iterations: int = 4000,
    anchor_loss_weight: float = 8.0,
    original_lr_scale: float = 0.05,
    dino_loss_weight: float = 0.05,
    depth_loss_weight: float = 0.05,
    device: str = "cuda",
) -> GaussianCloud:
    """
    Masked optimization of Gaussian parameters using gsplat.

    Loss = L1_photo (masked) + DINOv2_feature + Pearson_depth + anchor

    Args:
        cloud: GaussianCloud with mixed provenance
        target_images: List of [H, W, 3] float32 sRGB target images
        cameras: Corresponding cameras for each target image
        gap_masks: Optional [H, W] bool masks (True = gap region)
        iterations: Number of optimization iterations
        anchor_loss_weight: Weight for anchor loss
        original_lr_scale: Learning rate multiplier for original Gaussians
        dino_loss_weight: Weight for DINOv2 semantic feature loss
        depth_loss_weight: Weight for Pearson depth correlation loss
        device: Torch device

    Returns:
        Optimized GaussianCloud
    """
    from gsplat import rasterization

    if len(target_images) == 0:
        logger.warning("No target images for optimization, returning cloud unchanged")
        return cloud

    dev = torch.device(device)

    # Prepare parameters from cloud
    gsplat_data = cloud.to_gsplat()
    means = gsplat_data["means"].to(dev).requires_grad_(True)
    scales_raw = torch.log(gsplat_data["scales"].to(dev).clamp(min=1e-7))
    scales_raw.requires_grad_(True)
    quats = gsplat_data["quats"].to(dev).requires_grad_(True)
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

    # Target images in sRGB [0,1]
    targets = [torch.from_numpy(img).float().to(dev) for img in target_images]

    # Gap masks
    if gap_masks is not None:
        mask_tensors = [
            torch.from_numpy(m.astype(np.float32)).to(dev) for m in gap_masks
        ]
        has_masks = True
        mask_pct = np.mean([m.mean() for m in gap_masks]) * 100
    else:
        mask_tensors = None
        has_masks = False
        mask_pct = 100.0

    # Camera matrices
    K_list = [
        torch.from_numpy(cam.K).float().unsqueeze(0).to(dev) for cam in cameras
    ]
    viewmat_list = [
        torch.from_numpy(_opengl_w2c_to_opencv(cam.w2c)).float().unsqueeze(0).to(dev)
        for cam in cameras
    ]

    # Load DINOv2 for semantic feature loss
    use_dino = dino_loss_weight > 0
    if use_dino:
        dino_model = _get_dino_model(dev)
        # Pre-compute target DINOv2 features (frozen, no grad)
        target_dino_features = []
        with torch.no_grad():
            for t in targets:
                feat = _extract_dino_features(dino_model, t)
                target_dino_features.append(feat)

    n_views = len(targets)
    log_interval = max(iterations // 8, 100)

    logger.info(
        f"Optimization: {iterations} iters, {n_views} views, "
        f"{int(is_original.sum())} original + "
        f"{len(cloud) - int(is_original.sum())} seeded Gaussians"
    )
    logger.info(
        f"  Losses: L1_photo (masked {mask_pct:.0f}%) + "
        f"DINOv2 (w={dino_loss_weight}) + "
        f"depth_pearson (w={depth_loss_weight}) + "
        f"anchor (w={anchor_loss_weight})"
    )

    for step in range(iterations):
        optimizer.zero_grad()

        view_idx = step % n_views
        target = targets[view_idx]
        H, W = target.shape[:2]
        K = K_list[view_idx]
        viewmat = viewmat_list[view_idx]

        # Decode parameters
        scales = torch.exp(scales_raw)
        opacities = torch.sigmoid(opacities_logit)

        # Render RGB + expected depth
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
            render_mode="RGB+ED",
        )

        rendered_rgb = renders[0, :, :, :3]
        rendered_depth = renders[0, :, :, 3]

        # === Loss 1: L1 photometric (masked) ===
        pixel_error = (rendered_rgb - target).abs()
        if has_masks:
            mask = mask_tensors[view_idx].unsqueeze(-1)
            masked_error = pixel_error * mask
            n_masked = mask.sum().clamp(min=1)
            photo_loss = masked_error.sum() / (n_masked * 3)
        else:
            photo_loss = pixel_error.mean()

        # === Loss 2: DINOv2 semantic feature loss ===
        dino_loss = torch.tensor(0.0, device=dev)
        if use_dino and step % 5 == 0:  # Every 5th step (expensive)
            target_feat = target_dino_features[view_idx]
            rendered_feat = _extract_dino_features(dino_model, rendered_rgb)
            # Cosine similarity per patch, averaged
            sim = F.cosine_similarity(
                rendered_feat.reshape(-1, rendered_feat.shape[-1]),
                target_feat.reshape(-1, target_feat.shape[-1]),
                dim=-1,
            )
            dino_loss = (1 - sim.mean()) * dino_loss_weight

        # === Loss 3: Pearson depth correlation ===
        depth_loss = torch.tensor(0.0, device=dev)
        if depth_loss_weight > 0:
            # Use alpha-weighted mask: only where there's actual rendered content
            alpha_mask = alphas[0, :, :, 0] > 0.1
            if has_masks:
                depth_mask = alpha_mask & (mask_tensors[view_idx] > 0.5)
            else:
                depth_mask = alpha_mask

            if depth_mask.sum() > 100:
                # Target depth: estimate from rendered depth of non-gap regions
                # as reference, then correlate with gap-region depth
                rd = rendered_depth[depth_mask]
                # Self-consistency: depth should be smooth and correlated
                # across the gap boundary. Use rendered depth variance as proxy.
                # Pearson correlation with a smooth version of depth.
                if rd.numel() > 100:
                    # Smooth the depth with a small kernel
                    depth_2d = rendered_depth.unsqueeze(0).unsqueeze(0)
                    depth_smooth = F.avg_pool2d(
                        depth_2d, kernel_size=7, stride=1, padding=3
                    ).squeeze()
                    rd_smooth = depth_smooth[depth_mask]
                    pearson = _pearson_corrcoef(rd, rd_smooth)
                    depth_loss = (1 - pearson) * depth_loss_weight

        # === Loss 4: Anchor loss ===
        means_diff = (means[is_original] - anchor_means[is_original]).pow(2).sum(dim=-1).mean()
        colors_diff = (sh0_colors[is_original] - anchor_colors[is_original]).pow(2).sum(dim=-1).mean()
        anchor_loss = anchor_loss_weight * (means_diff + colors_diff)

        # Total loss
        loss = photo_loss + dino_loss + depth_loss + anchor_loss

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
                f"photo={photo_loss.item():.4f}, "
                f"dino={dino_loss.item():.4f}, "
                f"depth={depth_loss.item():.4f}, "
                f"anchor={anchor_loss.item():.4f}"
            )

    # Build output cloud
    with torch.no_grad():
        final_means = means.cpu().numpy()
        final_scales = torch.exp(scales_raw).cpu().numpy()
        final_quats_wxyz = (quats / quats.norm(dim=-1, keepdim=True)).cpu().numpy()
        final_opacities = torch.sigmoid(opacities_logit).cpu().numpy()
        final_colors_srgb = (sh0_colors * SH_C0 + 0.5).clamp(0, 1).cpu().numpy()
        final_quats = np.stack([
            final_quats_wxyz[:, 1],
            final_quats_wxyz[:, 2],
            final_quats_wxyz[:, 3],
            final_quats_wxyz[:, 0],
        ], axis=-1)

    logger.info(
        f"Optimization complete. Final: photo={photo_loss.item():.4f}, "
        f"dino={dino_loss.item():.4f}, depth={depth_loss.item():.4f}"
    )

    return GaussianCloud(
        means=final_means,
        scales=final_scales,
        quats=final_quats,
        colors=final_colors_srgb,
        opacities=final_opacities,
        provenance=cloud.provenance.copy(),
    )
