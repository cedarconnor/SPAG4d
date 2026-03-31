"""Shadow Gaussian validation: provisional → promoted lifecycle."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource
from ..camera.pinhole import PinholeCamera

logger = logging.getLogger(__name__)


def validate_shadow_gaussians(
    cloud: GaussianCloud,
    cameras: List[PinholeCamera],
    consistency_threshold: float = 0.8,
    synthesized_images: Optional[List[np.ndarray]] = None,
    color_consistency_threshold: float = 0.15,
    hallucination_opacity: float = 0.3,
) -> GaussianCloud:
    """
    Multi-view consistency check for shadow Gaussians.

    Projects each SEEDED Gaussian into multiple cameras to check visibility.
    Optionally checks color consistency across synthesized images to detect
    hallucinations.

    Args:
        cloud: GaussianCloud with SEEDED Gaussians
        cameras: Validation cameras
        consistency_threshold: Fraction of cameras for geometric promotion
        synthesized_images: Optional list of [H, W, 3] Klein outputs per camera.
            When provided, color consistency is checked for promoted Gaussians.
        color_consistency_threshold: Max L1 color distance between views
        hallucination_opacity: Reduced opacity for color-inconsistent Gaussians

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

    visibility_counts = np.zeros(n_seeded, dtype=np.int32)
    check_colors = (synthesized_images is not None and len(synthesized_images) == n_cams)

    if check_colors:
        sampled_colors = np.full((n_seeded, n_cams, 3), np.nan, dtype=np.float32)

    for cam_idx, cam in enumerate(cameras):
        w2c = cam.w2c
        pts_cam = (w2c[:3, :3] @ seeded_means.T + w2c[:3, 3:4]).T

        z = pts_cam[:, 2]
        in_front = z < -0.01

        neg_z = np.clip(-z, 0.01, None)
        u = pts_cam[:, 0] / neg_z * cam.fx + cam.cx
        v = cam.cy - pts_cam[:, 1] / neg_z * cam.fy

        in_bounds = (
            in_front
            & (u >= 0) & (u < cam.width)
            & (v >= 0) & (v < cam.height)
        )

        visibility_counts += in_bounds.astype(np.int32)

        if check_colors:
            visible_idx = np.where(in_bounds)[0]
            if len(visible_idx) > 0:
                px = np.clip(u[visible_idx].astype(int), 0, cam.width - 1)
                py = np.clip(v[visible_idx].astype(int), 0, cam.height - 1)
                synth = synthesized_images[cam_idx]
                sampled_colors[visible_idx, cam_idx] = synth[py, px]

    visibility_ratio = visibility_counts / max(n_cams, 1)
    promote = visibility_ratio >= consistency_threshold

    new_prov = cloud.provenance.copy()
    new_opacities = cloud.opacities.copy()
    seeded_indices = np.where(seeded_mask)[0]
    new_prov[seeded_indices[promote]] = GaussianSource.PROMOTED
    new_prov[seeded_indices[~promote]] = GaussianSource.PRUNED

    n_color_reduced = 0
    if check_colors:
        for local_idx in np.where(promote)[0]:
            colors = sampled_colors[local_idx]
            valid_colors = colors[~np.isnan(colors[:, 0])]
            if len(valid_colors) < 2:
                continue
            max_dist = 0.0
            for i in range(len(valid_colors)):
                for j in range(i + 1, len(valid_colors)):
                    dist = np.abs(valid_colors[i] - valid_colors[j]).mean()
                    max_dist = max(max_dist, dist)
            if max_dist > color_consistency_threshold:
                global_idx = seeded_indices[local_idx]
                new_opacities[global_idx] = hallucination_opacity
                n_color_reduced += 1

    n_promoted = int(np.sum(promote))
    n_pruned = n_seeded - n_promoted
    logger.info(
        f"Shadow validation: {n_seeded:,} seeded → "
        f"{n_promoted:,} promoted, {n_pruned:,} pruned "
        f"(threshold={consistency_threshold:.0%}, {n_cams} cameras)"
    )
    if check_colors and n_color_reduced > 0:
        logger.info(
            f"  Color consistency: {n_color_reduced:,} Gaussians had opacity "
            f"reduced to {hallucination_opacity} (threshold={color_consistency_threshold})"
        )

    cloud.provenance = new_prov
    cloud.opacities = new_opacities
    return cloud
