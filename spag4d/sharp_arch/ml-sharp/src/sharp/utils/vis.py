"""Utility functions for visualization.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt

METRIC_DEPTH_MAX_CLAMP_METER = 50.0


def colorize_depth(depth: torch.Tensor, val_max: float = 10.0) -> torch.Tensor:
    """Colorize depth map."""
    depth_channels = depth.shape[-3]

    # When we have a general depth/disparity map, output the color map as is.
    if depth_channels == 1:
        return colorize_scalar_map(
            depth.squeeze(-3), val_min=0.0, val_max=val_max, color_map="turbo"
        )

    # When we have a multi-layered depth/disparity map,
    # we concatenate the color maps horizontally and output it.
    else:
        colored_depths = []
        for c in range(depth_channels):
            colored_depths.append(
                colorize_scalar_map(
                    depth[..., c, :, :], val_min=0.0, val_max=val_max, color_map="turbo"
                )
            )
        return torch.cat(colored_depths, dim=-1)


def colorize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Colorize alpha map."""
    return colorize_scalar_map(alpha.squeeze(-3), val_min=0.0, val_max=1.0, color_map="coolwarm")


def colorize_scalar_map(
    scalar_map: torch.Tensor, val_min=0.0, val_max=1.0, color_map: str = "jet"
) -> torch.Tensor:
    """Colorize a scalar map of.

    Args:
        scalar_map: Map of with format BHW.
        val_min: Minimu value to display.
        val_max: Maximum value to display.
        color_map: Which color map to use. Will be passed to matplotlob.

    Returns:
        A colorized image with format BHWC.
    """
    if scalar_map.ndim not in (2, 3, 4):
        raise ValueError("Only scalar maps of 2 or 3 or 4 dimensions supported.")

    cmap = plt.get_cmap(color_map)

    scalar_map_np = scalar_map.detach().cpu().float().numpy()
    scalar_map_np = (scalar_map_np - val_min) / (val_max - val_min)
    scalar_map_np = np.clip(scalar_map_np, a_min=0.0, a_max=1.0)

    color_map_np = cmap(scalar_map_np)[..., :3]
    tensor = torch.as_tensor(color_map_np * 255.0, dtype=torch.uint8)

    if tensor.ndim == 3:
        return tensor.permute(2, 0, 1)
    elif tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:
        return tensor.permute(0, 1, 4, 2, 3)
    else:
        assert False, "Invalid tensor shape encountered."
