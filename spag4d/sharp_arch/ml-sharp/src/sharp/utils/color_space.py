"""Contains color space utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from sharp.utils.robust import robust_where

LOGGER = logging.getLogger(__name__)

ColorSpace = Literal["sRGB", "linearRGB"]


def encode_color_space(color_space: ColorSpace) -> int:
    """Encode color space to integer."""
    return 0 if color_space == "sRGB" else 1


def decode_color_space(color_space_index: int) -> ColorSpace:
    """Decode color space index to color space."""
    return "sRGB" if color_space_index == 0 else "linearRGB"


def sRGB2linearRGB(sRGB: torch.Tensor) -> torch.Tensor:
    """SRGB to linearRGB conversion function.

    Reference:
    https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    Section 7.7.7

    Args:
        sRGB: Input image tensor in sRGB space.
    """
    # We need to use robust_where to clamp the second branch.
    # Otherwise, torch.where will lead to NaN in the backward pass, see
    # https://github.com/pytorch/pytorch/issues/68425
    THRESHOLD = 0.04045

    def branch_true_func(x):
        return x / 12.92

    def branch_false_func(x):
        return ((x + 0.055) / 1.055) ** 2.4

    return robust_where(
        sRGB <= THRESHOLD,
        sRGB,
        branch_true_func,
        branch_false_func,
        branch_false_safe_value=THRESHOLD,
    )


def linearRGB2sRGB(linearRGB: torch.Tensor) -> torch.Tensor:
    """LinearRGB to sRGB conversion function.

    Reference:
    https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    Section 7.7.7

    Args:
        linearRGB: Input image tensor in linearRGB space.
    """
    # We need to use robust_where to clamp the second branch.
    # Otherwise, torch.where will lead to NaN in the backward pass, see
    # https://github.com/pytorch/pytorch/issues/68425
    THRESHOLD = 0.0031308

    def branch_true_func(x):
        return x * 12.92

    def branch_false_func(x):
        return 1.055 * (x ** (1 / 2.4)) - 0.055

    return robust_where(
        linearRGB <= THRESHOLD,
        linearRGB,
        branch_true_func,
        branch_false_func,
        branch_false_safe_value=THRESHOLD,
    )
