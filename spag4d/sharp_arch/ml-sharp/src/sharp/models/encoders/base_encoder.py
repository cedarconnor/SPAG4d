"""Contains the base class for encoders.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import abc

import torch
from torch import nn


class BaseEncoder(nn.Module, abc.ABC):
    """Base encoder class."""

    dim_in: int
    output_dims: list[int]

    @abc.abstractmethod
    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Encode input image into multi-resolution encodings."""

    def internal_resolution(self) -> int:
        """Internal resolution of the encoder."""
        return 1536
