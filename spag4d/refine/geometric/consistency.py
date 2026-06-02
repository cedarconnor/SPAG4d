# spag4d/refine/geometric/consistency.py
"""Cross-frame mutual support gate — rejects single-frame inventions."""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial import cKDTree

from .hole_filter import FilterResult


@dataclass
class ConsistencyConfig:
    min_support_count: int = 2
    support_radius: float | None = None   # None -> set later from voxel_size
    positional_std_max: float | None = None
    strict_mode_for_disoccl: bool = True


def cross_frame_consistency_gate(
    per_frame_results: List[FilterResult],
    config: ConsistencyConfig,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """Filter all per-frame candidates by multi-frame support.

    Args:
        per_frame_results: one FilterResult per OmniRoam frame.
        config: consistency thresholds.
        voxel_size: used to set support_radius if not specified.

    Returns:
        (K, 3) float32 surviving points with provenance arrays attached as
        a structured numpy array with fields: x, y, z, frame_idx, hole_mode_int.
    """
    if not per_frame_results:
        return np.zeros((0, 3), dtype=np.float32)

    radius = config.support_radius if config.support_radius is not None else 1.5 * voxel_size

    # Concatenate all candidates
    all_pts = []
    all_frame_idx = []
    all_mode = []  # 0=alpha, 1=disoccl
    for r in per_frame_results:
        if r.num_kept == 0:
            continue
        all_pts.append(r.points)
        all_frame_idx.extend([r.source_frame_idx] * r.num_kept)
        all_mode.extend([0 if m == "alpha" else 1 for m in r.hole_modes])

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)

    pts = np.vstack(all_pts).astype(np.float32)
    frame_idx = np.array(all_frame_idx, dtype=np.int32)
    mode = np.array(all_mode, dtype=np.uint8)

    tree = cKDTree(pts)
    keep = np.zeros(len(pts), dtype=bool)

    for i, pt in enumerate(pts):
        neighbor_indices = tree.query_ball_point(pt, r=radius)
        # Count distinct frames in the neighborhood INCLUDING own frame
        neighbor_frames = set(int(frame_idx[j]) for j in neighbor_indices)
        support = len(neighbor_frames)

        required = config.min_support_count
        if mode[i] == 1 and config.strict_mode_for_disoccl:
            required += 1

        if support < required:
            continue

        if config.positional_std_max is not None and len(neighbor_indices) > 1:
            neighbor_pts = pts[neighbor_indices]
            std = float(np.std(neighbor_pts))
            if std > config.positional_std_max:
                continue

        keep[i] = True

    return pts[keep]
