"""Depth → grayscale disparity image for Klein conditioning input.

Klein expects a grayscale depth map: white = close, black = far.
This matches the standard depth conditioning format used by FLUX/Klein
models (see: civitai.com/models/2349427).
"""

from __future__ import annotations

import numpy as np


def depth_to_disparity_image(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    **kwargs,
) -> np.ndarray:
    """
    Convert forward-warped depth to a grayscale disparity image.

    Produces white (close) to black (far) grayscale suitable for
    Klein/FLUX depth conditioning. Invalid pixels are black.

    Args:
        depth: [H, W] Z-depth in meters (inf at holes)
        valid_mask: [H, W] bool
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        [H, W, 3] float32 [0, 1] grayscale depth visualization
    """
    H, W = depth.shape
    result = np.zeros((H, W, 3), dtype=np.float32)

    if not valid_mask.any():
        return result

    # Compute disparity at valid pixels (1/depth: close=high, far=low)
    valid_depth = depth[valid_mask]
    disparity = 1.0 / np.clip(valid_depth, 1e-3, None)

    # Percentile clipping for dynamic range
    vmin = np.percentile(disparity, percentile_low)
    vmax = np.percentile(disparity, percentile_high)

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    # Normalize to [0, 1]: close=1 (white), far=0 (black)
    disp_norm = np.zeros((H, W), dtype=np.float32)
    disp_norm[valid_mask] = np.clip((disparity - vmin) / (vmax - vmin), 0, 1)

    # Grayscale: replicate to 3 channels
    result[valid_mask, 0] = disp_norm[valid_mask]
    result[valid_mask, 1] = disp_norm[valid_mask]
    result[valid_mask, 2] = disp_norm[valid_mask]

    return result
