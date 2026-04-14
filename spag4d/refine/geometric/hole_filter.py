# spag4d/refine/geometric/hole_filter.py
"""Per-frame alpha-gate and disocclusion-gate hole filtering."""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .depth_convention import is_nearer_than_rendered


@dataclass
class HoleFilterConfig:
    alpha_hole_threshold: float = 0.30
    alpha_confident_threshold: float = 0.90
    depth_disoccl_margin_ratio: float = 0.02
    local_median_window: int = 5


@dataclass
class FilterResult:
    points: np.ndarray        # (K, 3)
    candidate_z: np.ndarray   # (K,)
    hole_modes: List[str]     # "alpha" | "disoccl"
    source_frame_idx: int
    src_uv: np.ndarray        # (K, 2) int, pixel of origin

    @property
    def num_kept(self) -> int:
        return len(self.points)


def filter_candidate_points_per_frame(
    candidates: np.ndarray,
    candidate_z: np.ndarray,
    source_frame_idx: int,
    rendered_depth: np.ndarray,
    rendered_alpha: np.ndarray,
    src_uv: np.ndarray,
    config: HoleFilterConfig,
) -> FilterResult:
    """Filter per-frame unprojected candidate points to genuine holes.

    Args:
        candidates: (K, 3) world-space candidate points.
        candidate_z: (K,) camera-forward z-depth for each candidate.
        source_frame_idx: frame index for provenance.
        rendered_depth: (H, W) rendered base splat z-depth.
        rendered_alpha: (H, W) rendered base splat alpha.
        src_uv: (K, 2) int array of (u, v) pixel coords in frame space.
        config: filter thresholds.

    Returns:
        FilterResult with kept points only.
    """
    H, W = rendered_alpha.shape
    u = np.clip(src_uv[:, 0], 0, W - 1)
    v = np.clip(src_uv[:, 1], 0, H - 1)

    alpha_at_pt = rendered_alpha[v, u]
    depth_at_pt = rendered_depth[v, u]

    keep_mask = np.zeros(len(candidates), dtype=bool)
    modes = [""] * len(candidates)

    for i in range(len(candidates)):
        a = alpha_at_pt[i]
        dz = candidate_z[i]
        dr = depth_at_pt[i]

        if a < config.alpha_hole_threshold:
            keep_mask[i] = True
            modes[i] = "alpha"
        elif a < config.alpha_confident_threshold:
            # Compute local median depth in window
            hw = config.local_median_window // 2
            v0 = max(0, int(v[i]) - hw)
            v1 = min(H, int(v[i]) + hw + 1)
            u0 = max(0, int(u[i]) - hw)
            u1 = min(W, int(u[i]) + hw + 1)
            local_med = float(np.median(rendered_depth[v0:v1, u0:u1]))
            if is_nearer_than_rendered(
                np.array([dz]),
                np.array([dr]),
                margin_ratio=config.depth_disoccl_margin_ratio,
                local_depth_scale=np.array([local_med]),
            )[0]:
                keep_mask[i] = True
                modes[i] = "disoccl"

    kept_idx = np.where(keep_mask)[0]
    return FilterResult(
        points=candidates[kept_idx],
        candidate_z=candidate_z[kept_idx],
        hole_modes=[modes[i] for i in kept_idx],
        source_frame_idx=source_frame_idx,
        src_uv=src_uv[kept_idx],
    )
