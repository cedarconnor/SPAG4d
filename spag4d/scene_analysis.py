"""Auto-compute scene-relative parameters from depth map statistics."""

from __future__ import annotations

import numpy as np


def compute_scene_defaults(
    depth_map: np.ndarray,
    image_height: int | None = None,
) -> dict:
    """
    Compute scale-relative pipeline defaults from a depth map.

    Works for any scene scale — a 3m room and a 100m forest both
    get appropriate thresholds without manual tuning.

    Args:
        depth_map: [H, W] depth in meters (0 or negative = invalid)
        image_height: Image height for pixel-relative params.
            Defaults to depth_map.shape[0].

    Returns:
        Dict with keys: sky_threshold, depth_min, depth_max,
        orbit_radius, confidence_decay_pixels
    """
    if image_height is None:
        image_height = depth_map.shape[0]

    # Mask out invalid depths (zero, negative, inf)
    valid = (depth_map > 0.01) & np.isfinite(depth_map)
    if valid.sum() < 100:
        # Fallback for degenerate depth maps
        return {
            "sky_threshold": 80.0,
            "depth_min": 0.1,
            "depth_max": 100.0,
            "orbit_radius": 0.5,
            "confidence_decay_pixels": max(10, int(image_height * 0.02)),
        }

    valid_depths = depth_map[valid]

    p1 = float(np.percentile(valid_depths, 1))
    p50 = float(np.percentile(valid_depths, 50))
    p95 = float(np.percentile(valid_depths, 95))
    p99 = float(np.percentile(valid_depths, 99))

    return {
        "sky_threshold": max(p95, p1 + 1.0),  # At least 1m range
        "depth_min": max(0.01, p1 * 0.8),     # 20% margin below 1st percentile
        "depth_max": p99 * 1.1,                # 10% margin above 99th percentile
        "orbit_radius": max(0.05, p50 * 0.05), # 5% of median depth
        "confidence_decay_pixels": max(10, int(image_height * 0.02)),
    }
