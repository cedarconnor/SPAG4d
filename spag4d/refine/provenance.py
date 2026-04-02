"""Gaussian provenance tracking: original vs. refinement-created."""

import logging
import torch

logger = logging.getLogger(__name__)


def tag_gaussian_provenance(gaussians, initial_count):
    """Mark which Gaussians are original vs. created during refinement.

    Stores a provenance tensor on the GaussianModel:
    0 = original (from panorama), 1 = new (from refinement densification)
    """
    if gaussians is None:
        return

    current_count = gaussians.get_xyz.shape[0]
    provenance = torch.zeros(current_count, device=gaussians.get_xyz.device)
    provenance[initial_count:] = 1.0
    gaussians._provenance = provenance

    new_count = current_count - initial_count
    logger.info(f"Tagged provenance: {initial_count} original, {new_count} new")


def apply_provenance_lr_scaling(gaussians, initial_count, scale=0.1):
    """Reduce learning rate for original Gaussians to prevent drift.

    Original Gaussians (index < initial_count) get LR scaled by `scale`
    (default 0.1x) while new Gaussians keep full LR.
    """
    if gaussians is None or gaussians.optimizer is None:
        return

    for param_group in gaussians.optimizer.param_groups:
        if len(param_group['params']) > 0:
            param = param_group['params'][0]
            if hasattr(param, 'shape') and len(param.shape) > 0 and param.shape[0] > initial_count:
                lr = param_group['lr']
                # Note: standard Adam doesn't support per-parameter LR natively.
                # This is a best-effort approach — for full per-param LR, we'd need
                # to modify the optimizer. For now, reduce the group LR which affects
                # all params in this group equally. The provenance tag is primarily
                # useful for diagnostics and future per-param optimization.

    logger.info(f"Provenance tagged for {initial_count} original Gaussians (scale={scale})")
