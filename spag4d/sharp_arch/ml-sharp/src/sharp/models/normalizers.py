"""Contains an implementation of image normalizers for perceptual loss.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import nn


class MeanStdNormalizer(nn.Module):
    """Normalizing image input by mean and std."""

    mean: torch.Tensor
    std_inv: torch.Tensor

    def __init__(
        self,
        mean: Union[Sequence[float], torch.Tensor],
        std: Union[Sequence[float], torch.Tensor],
    ):
        """Initialize MeanStdNormalizer."""
        super(MeanStdNormalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean).view(-1, 1, 1)
        if not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std).view(-1, 1, 1)
        self.register_buffer("mean", mean)
        # We use inverse std to use a multiplication which is better supported by the hardware
        self.register_buffer("std_inv", 1.0 / std)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply mean and std normalization over input image."""
        return (image - self.mean) * self.std_inv


class AffineRangeNormalizer(nn.Module):
    """Perform linear mapping to map input_range to output_range.

    Output_range defaults to (0, 1).
    """

    def __init__(
        self,
        input_range: tuple[float, float],
        output_range: tuple[float, float] = (0, 1),
    ):
        """Initialize AffineRangeNormalizer."""
        super().__init__()
        input_min, input_max = input_range
        output_min, output_max = output_range
        if input_max <= input_min:
            raise ValueError(f"Invalid input_range: {input_range}")
        if output_max <= output_min:
            raise ValueError(f"Invalid output_range: {output_range}")

        self.scale = (output_max - output_min) / (input_max - input_min)
        self.bias = output_min - input_min * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine range normalization over input image."""
        if self.scale != 1.0:
            x = x * self.scale

        if self.bias != 0.0:
            x = x + self.bias

        return x


class MobileNetNormalizer(AffineRangeNormalizer):
    """Image normalization in mobilenet."""

    def __init__(self, input_range: tuple[float, float] = (0, 1)):
        """Initialize MobileNetNormalizer."""
        super().__init__(input_range=input_range, output_range=(-1, 1))
