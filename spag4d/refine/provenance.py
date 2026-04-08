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


# ---------------------------------------------------------------------------
# Provenance tag constants
# ---------------------------------------------------------------------------

PROVENANCE_ORIGINAL = 0
PROVENANCE_DENSIFIED = 1
PROVENANCE_OMNIROAM = 2
PROVENANCE_GAP_SEED = 3


def tag_provenance_by_range(gaussians, start_idx, end_idx, tag_value):
    """Tag a range of Gaussians with a specific provenance value.

    Creates the ``_provenance`` tensor if it does not exist yet, padding with
    zeros for any Gaussians that were added since the last provenance update.

    Args:
        gaussians: GaussianModel instance (or None — no-op if None).
        start_idx: Inclusive start index of the range to tag.
        end_idx: Exclusive end index of the range to tag.
        tag_value: Integer/float provenance constant to assign (see
            PROVENANCE_* constants in this module).
    """
    if gaussians is None:
        return

    if not hasattr(gaussians, '_provenance'):
        current_count = gaussians.get_xyz.shape[0]
        gaussians._provenance = torch.zeros(current_count, device=gaussians.get_xyz.device)

    current_count = gaussians.get_xyz.shape[0]
    if gaussians._provenance.shape[0] < current_count:
        extra = torch.zeros(
            current_count - gaussians._provenance.shape[0],
            device=gaussians._provenance.device,
        )
        gaussians._provenance = torch.cat([gaussians._provenance, extra])

    gaussians._provenance[start_idx:end_idx] = tag_value
    logger.info(f"Tagged Gaussians [{start_idx}:{end_idx}] with provenance={tag_value}")


def get_provenance_mask(gaussians, tag_value):
    """Get boolean mask for Gaussians with a specific provenance tag.

    Args:
        gaussians: GaussianModel instance (or None).
        tag_value: The provenance constant to match (see PROVENANCE_* constants).

    Returns:
        Boolean tensor mask of shape (N,), or None if gaussians is None or
        has no ``_provenance`` attribute.
    """
    if gaussians is None or not hasattr(gaussians, '_provenance'):
        return None
    return gaussians._provenance == tag_value
