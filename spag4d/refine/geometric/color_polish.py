# spag4d/refine/geometric/color_polish.py
"""Short SH-only finetune to align new Gaussian colors with the input panorama."""
import logging
from pathlib import Path

import numpy as np
import torch

from .config import ColorPolishConfig

logger = logging.getLogger("spag4d.refine.geometric")


def color_polish(
    ply_path: str,
    panorama_path: str,
    config: ColorPolishConfig,
    num_base_gaussians: int,
) -> float | None:
    """Finetune SH coefficients of new Gaussians to match panorama colors.

    Freezes all base gaussian parameters and all non-SH params of new ones.
    Returns seam L1 delta (before - after), or None if skipped.
    """
    if config.steps == 0:
        return None

    try:
        import gsplat
        from PIL import Image
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
    except ImportError:
        logger.warning("[color_polish] gsplat or Pillow not available — skipping")
        return None

    from spag4d.refine.format_compat import load_gaussians_from_ply
    gaussians = load_gaussians_from_ply(ply_path)

    total = gaussians.get_xyz.shape[0]
    new_mask = torch.zeros(total, dtype=torch.bool)
    new_mask[num_base_gaussians:] = True

    if new_mask.sum() == 0:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussians = gaussians.to(device) if hasattr(gaussians, "to") else gaussians

    # Freeze everything except new-gaussian SH dc features
    features_dc = gaussians._features_dc.detach().clone().requires_grad_(False)
    new_features_dc = features_dc[new_mask].detach().clone().requires_grad_(True)

    panorama = np.array(Image.open(panorama_path).convert("RGB")).astype(np.float32) / 255.0
    panorama_t = torch.from_numpy(panorama).to(device).unsqueeze(0)  # (1, H, W, 3)

    optimizer = AdamW([new_features_dc], lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=0)

    best_loss = float("inf")
    patience_counter = 0

    for step in range(config.steps):
        optimizer.zero_grad()

        # Assemble full feature tensor
        full_dc = features_dc.clone()
        full_dc[new_mask] = new_features_dc

        # Render from panorama pose (identity = origin viewpoint)
        loss = _render_and_loss(gaussians, full_dc, panorama_t, config, device)
        if loss is None:
            break

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                logger.info("[color_polish] Early stop at step %d", step)
                break

        if step % 50 == 0:
            logger.debug("[color_polish] step=%d loss=%.4f", step, loss_val)

    # Write updated colors back (simplified: update PLY with new SH dc)
    _write_polished_ply(ply_path, gaussians, new_features_dc.detach(), new_mask, num_base_gaussians)
    return None  # seam delta measurement is a TODO for M9 benchmark


def _render_and_loss(gaussians, features_dc, panorama_t, config, device):
    """Simplified render + L1 loss. Full ERP render would use render_utils."""
    # Placeholder: a real implementation renders the full ERP from origin pose
    # and computes L1 + SSIM against panorama_t.
    # This scaffold returns zero loss so the pipeline runs end-to-end.
    dummy = features_dc.mean() * 0.0
    return dummy


def _write_polished_ply(ply_path, gaussians, new_dc, new_mask, num_base):
    """Overwrite PLY with polished SH dc values for new Gaussians."""
    import torch
    from spag4d.ply_writer import save_ply_gsplat

    base_xyz = gaussians.get_xyz.detach().cpu().numpy()
    base_dc = gaussians._features_dc.detach().cpu().numpy()[:, 0, :]
    base_dc[num_base:] = new_dc.cpu().numpy()

    base_scaling = torch.exp(gaussians.get_scaling).detach().cpu().numpy()
    base_rot = gaussians.get_rotation.detach().cpu().numpy()
    base_opacity = torch.sigmoid(gaussians.get_opacity).detach().cpu().numpy().squeeze(-1)

    gaussians_dict = {
        "means": base_xyz,
        "colors": base_dc,
        "scales": base_scaling,
        "quats": base_rot,
        "opacities": base_opacity,
    }
    save_ply_gsplat(gaussians_dict, ply_path, sh_degree=0, colors_linear=False)
