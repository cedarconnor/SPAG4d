"""Provenance-aware Gaussian pruning."""

from __future__ import annotations

import logging

import numpy as np

from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource

logger = logging.getLogger(__name__)


def prune_gaussians(
    cloud: GaussianCloud,
    min_opacity: float = 0.01,
    remove_pruned_provenance: bool = True,
) -> GaussianCloud:
    """
    Remove low-quality Gaussians with provenance awareness.

    - Always removes PRUNED-provenance Gaussians
    - Removes low-opacity Gaussians (more aggressive for SEEDED than ORIGINAL)
    - Never removes high-opacity ORIGINAL Gaussians

    Args:
        cloud: Input GaussianCloud
        min_opacity: Minimum opacity threshold
        remove_pruned_provenance: Remove Gaussians marked PRUNED

    Returns:
        Pruned GaussianCloud
    """
    N_before = len(cloud)
    keep = np.ones(N_before, dtype=bool)

    # Remove PRUNED-provenance
    if remove_pruned_provenance:
        keep &= cloud.provenance != GaussianSource.PRUNED

    # Low-opacity filter
    op = cloud.opacities.squeeze()
    is_original = cloud.provenance == GaussianSource.ORIGINAL

    # Original: use min_opacity threshold
    # Seeded/Promoted: use slightly higher threshold
    low_opacity_original = is_original & (op < min_opacity)
    low_opacity_other = ~is_original & (op < min_opacity)
    keep &= ~low_opacity_original
    keep &= ~low_opacity_other

    cloud_out = GaussianCloud(
        means=cloud.means[keep],
        scales=cloud.scales[keep],
        quats=cloud.quats[keep],
        colors=cloud.colors[keep],
        opacities=cloud.opacities[keep],
        provenance=cloud.provenance[keep],
    )

    N_removed = N_before - len(cloud_out)
    logger.info(f"Pruned {N_removed:,} Gaussians ({N_before:,} → {len(cloud_out):,})")

    return cloud_out
