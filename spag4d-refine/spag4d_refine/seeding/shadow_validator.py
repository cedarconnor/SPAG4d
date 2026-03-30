"""Shadow Gaussian validation: provisional → promoted lifecycle."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource
from ..camera.pinhole import PinholeCamera

logger = logging.getLogger(__name__)


def validate_shadow_gaussians(
    cloud: GaussianCloud,
    cameras: List[PinholeCamera],
    consistency_threshold: float = 0.8,
) -> GaussianCloud:
    """
    Multi-view consistency check for shadow Gaussians.

    Projects each SEEDED Gaussian into multiple cameras to check if it's
    visible and consistent. Promotes consistent ones, discards outliers.

    Args:
        cloud: GaussianCloud with SEEDED Gaussians
        cameras: Validation cameras
        consistency_threshold: Fraction of cameras where the Gaussian
            must project into valid screen space

    Returns:
        Updated GaussianCloud with SEEDED → PROMOTED or PRUNED
    """
    seeded_mask = cloud.provenance == GaussianSource.SEEDED
    n_seeded = int(np.sum(seeded_mask))

    if n_seeded == 0:
        return cloud

    seeded_means = cloud.means[seeded_mask]
    n_cams = len(cameras)

    if n_cams == 0:
        logger.warning("No validation cameras, promoting all seeded Gaussians")
        new_prov = cloud.provenance.copy()
        new_prov[seeded_mask] = GaussianSource.PROMOTED
        cloud.provenance = new_prov
        return cloud

    # For each seeded Gaussian, count how many cameras it projects into
    visibility_counts = np.zeros(n_seeded, dtype=np.int32)

    for cam in cameras:
        w2c = cam.w2c
        pts_cam = (w2c[:3, :3] @ seeded_means.T + w2c[:3, 3:4]).T

        # OpenGL convention: camera looks along -Z, so points in front
        # have negative Z in camera space
        z = pts_cam[:, 2]
        in_front = z < -0.01

        # Project to pixel coords (OpenGL: divide by -z for positive depth)
        neg_z = np.clip(-z, 0.01, None)
        u = pts_cam[:, 0] / neg_z * cam.fx + cam.cx
        v = cam.cy - pts_cam[:, 1] / neg_z * cam.fy

        in_bounds = (
            in_front
            & (u >= 0) & (u < cam.width)
            & (v >= 0) & (v < cam.height)
        )

        visibility_counts += in_bounds.astype(np.int32)

    # Promote if visible in enough cameras
    visibility_ratio = visibility_counts / max(n_cams, 1)
    promote = visibility_ratio >= consistency_threshold

    new_prov = cloud.provenance.copy()
    seeded_indices = np.where(seeded_mask)[0]
    new_prov[seeded_indices[promote]] = GaussianSource.PROMOTED
    new_prov[seeded_indices[~promote]] = GaussianSource.PRUNED

    n_promoted = int(np.sum(promote))
    n_pruned = n_seeded - n_promoted
    logger.info(
        f"Shadow validation: {n_seeded:,} seeded → "
        f"{n_promoted:,} promoted, {n_pruned:,} pruned "
        f"(threshold={consistency_threshold:.0%}, {n_cams} cameras)"
    )

    cloud.provenance = new_prov
    return cloud
