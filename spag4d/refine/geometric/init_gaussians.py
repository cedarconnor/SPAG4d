# spag4d/refine/geometric/init_gaussians.py
"""New Gaussian initialization for hole-fill injection."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from .aggregate import AggregatedCandidates


@dataclass
class GaussianProvenance:
    source: Literal["base_panorama", "omniroam_geometric"]
    primary_source_frame_idx: int | None = None
    support_count: int = 0
    alignment_scale: float | None = None
    alignment_shift: float | None = None
    hole_mode: Literal["alpha", "disoccl"] | None = None
    positional_std: float | None = None


def new_gaussian_dict(
    aggregated: AggregatedCandidates,
    knn_scale: float = 0.75,
    initial_opacity: float = 0.35,
    k_neighbors: int = 5,
) -> dict:
    """Build a dict of new Gaussian parameters matching ply_writer.save_ply_gsplat format.

    Returns dict with keys: means, scales, quats, colors, opacities.
    All as float32 numpy arrays.

    Scale heuristic: k * median kNN distance (isotropic), matching DA360 generator.
    """
    pts = aggregated.positions  # (M, 3)
    M = len(pts)

    # kNN distance for scale estimation
    if M > k_neighbors:
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k_neighbors + 1)  # +1 to exclude self
        knn_dist = dists[:, 1:].mean(axis=1)  # (M,) mean of k nearest
    else:
        knn_dist = np.full(M, aggregated.voxel_size, dtype=np.float32)

    scale = (knn_scale * knn_dist).astype(np.float32)
    scales = np.stack([scale, scale, scale], axis=-1)  # isotropic (M, 3)

    # Identity quaternion WXYZ
    quats = np.zeros((M, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # W=1

    # Opacity in logit space: logit(0.35) = log(0.35/0.65)
    logit_opacity = float(np.log(initial_opacity / (1.0 - initial_opacity)))
    opacities = np.full(M, logit_opacity, dtype=np.float32)

    # Colors: aggregated RGB already [0, 1] sRGB
    colors = aggregated.colors.astype(np.float32)

    return {
        "means": pts,
        "scales": scales,
        "quats": quats,
        "colors": colors,
        "opacities": opacities,
    }


def initialize_hole_gaussians(
    aggregated: AggregatedCandidates,
    knn_scale: float = 0.75,
    initial_opacity: float = 0.35,
) -> dict:
    """Alias to new_gaussian_dict for API consistency."""
    return new_gaussian_dict(aggregated, knn_scale=knn_scale, initial_opacity=initial_opacity)


def estimate_base_voxel_size(base_xyz: np.ndarray, k: int = 5) -> float:
    """Estimate local Gaussian spacing in base splat via kNN."""
    if len(base_xyz) < k + 1:
        return 0.05
    sample = base_xyz if len(base_xyz) <= 50_000 else base_xyz[
        np.random.choice(len(base_xyz), 50_000, replace=False)
    ]
    tree = cKDTree(sample)
    dists, _ = tree.query(sample, k=k + 1)
    return float(np.median(dists[:, 1:]))
