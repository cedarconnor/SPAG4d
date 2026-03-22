"""Depth → disparity colormap visualization for Klein conditioning input."""

from __future__ import annotations

import numpy as np


def depth_to_disparity_image(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    colormap: str = "turbo",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """
    Convert forward-warped depth to a disparity colormap image.

    Disparity (1/depth) emphasizes foreground-background separation.
    Invalid pixels (disocclusion holes) are rendered as black.

    Args:
        depth: [H, W] Z-depth in meters (inf at holes)
        valid_mask: [H, W] bool
        colormap: matplotlib colormap name
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        [H, W, 3] float32 [0, 1] colormap visualization
    """
    try:
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError(
            "matplotlib is required for depth visualization. "
            "Install with: pip install spag4d-refine[synthesis]"
        )

    H, W = depth.shape
    result = np.zeros((H, W, 3), dtype=np.float32)

    if not valid_mask.any():
        return result

    # Compute disparity at valid pixels
    valid_depth = depth[valid_mask]
    disparity = 1.0 / np.clip(valid_depth, 1e-3, None)

    # Percentile clipping
    vmin = np.percentile(disparity, percentile_low)
    vmax = np.percentile(disparity, percentile_high)

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    # Normalize to [0, 1]
    disp_full = np.zeros((H, W), dtype=np.float32)
    disp_full[valid_mask] = np.clip((disparity - vmin) / (vmax - vmin), 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(disp_full)[..., :3].astype(np.float32)  # [H, W, 3]

    # Black out invalid pixels
    result[valid_mask] = colored[valid_mask]

    return result
