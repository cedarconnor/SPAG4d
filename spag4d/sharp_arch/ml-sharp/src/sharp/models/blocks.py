"""Contains reusable network components.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

NormLayerName = Literal["noop", "batch_norm", "group_norm", "instance_norm"]
UpsamplingMode = Literal["transposed_conv", "nearest", "bilinear"]


def norm_layer_2d(num_features: int, norm_type: NormLayerName, num_groups: int = 8) -> nn.Module:
    """Create normalization layer."""
    if norm_type == "noop":
        return nn.Identity()
    elif norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features=num_features)
    elif norm_type == "group_norm":
        return nn.GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type == "instance_norm":
        return nn.InstanceNorm2d(num_features=num_features)
    else:
        raise ValueError(f"Invalid normalization layer type: {norm_type}")


def upsampling_layer(upsampling_mode: UpsamplingMode, scale_factor: int, dim_in: int) -> nn.Module:
    """Create upsampling layer."""
    if upsampling_mode == "transposed_conv":
        return nn.ConvTranspose2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=scale_factor,
            stride=scale_factor,
            padding=0,
            bias=False,
        )
    elif upsampling_mode in ("nearest", "bilinear"):
        return nn.Upsample(scale_factor=scale_factor, mode=upsampling_mode)
    else:
        raise ValueError(f"Invalid upsampling mode {upsampling_mode}.")


class ResidualBlock(nn.Module):
    """Generic implementation of residual blocks.

    This implements a generic residual block from

        He et al. - Identity Mappings in Deep Residual Networks (2016),
        https://arxiv.org/abs/1603.05027

    which can be further customized via factory functions.
    """

    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        """Initialize ResidualBlock."""
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        delta_x = self.residual(x)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + delta_x


def residual_block_2d(
    dim_in: int,
    dim_out: int,
    dim_hidden: int | None = None,
    actvn: nn.Module | None = None,
    norm_type: NormLayerName = "noop",
    norm_num_groups: int = 8,
    dilation: int = 1,
    kernel_size: int = 3,
):
    """Create a simple 2D residual block."""
    if actvn is None:
        actvn = nn.ReLU()

    if dim_hidden is None:
        dim_hidden = dim_out // 2

    # Padding to maintain output size
    # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    padding = (dilation * (kernel_size - 1)) // 2

    def _create_block(dim_in: int, dim_out: int) -> list[nn.Module]:
        layers = [
            norm_layer_2d(dim_in, norm_type, num_groups=norm_num_groups),
            actvn,
        ]

        layers.append(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
            )
        )
        return layers

    residual = nn.Sequential(
        *_create_block(dim_in, dim_hidden),
        *_create_block(dim_hidden, dim_out),
    )
    shortcut = None

    if dim_in != dim_out:
        shortcut = nn.Conv2d(dim_in, dim_out, 1)

    return ResidualBlock(residual, shortcut)


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion for DPT."""

    # We use the name "deconv" for backward compatibility. However, "deconv" can also
    # refer to some other upsampling layer or a no-op.
    deconv: nn.Module

    def __init__(
        self,
        dim_in: int,
        dim_out: int | None = None,
        upsampling_mode: UpsamplingMode | None = None,
        batch_norm: bool = False,
    ):
        """Initialize feature fusion block.

        Args:
            dim_in: Dimensions of input.
            dim_out: Dimensions of output.
            batch_norm: Whether to use batch normalization in resnet blocks.
            upsampling_mode: What mode to use for upsampling. None if no upsampling
                is required.
        """
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.resnet1 = self._residual_block(dim_in, batch_norm)
        self.resnet2 = self._residual_block(dim_in, batch_norm)

        if upsampling_mode is not None:
            self.deconv = upsampling_layer(upsampling_mode, scale_factor=2, dim_in=dim_in)
        else:
            self.deconv = nn.Sequential()

        self.out_conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        """Process and fuse input features."""
        x = x0

        if x1 is not None:
            res = self.resnet1(x1)
            x = self.skip_add.add(x, res)

        x = self.resnet2(x)
        x = self.deconv(x)
        x = self.out_conv(x)

        return x

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool):
        """Create a residual block."""

        def _create_block(dim: int, batch_norm: bool) -> list[nn.Module]:
            layers = [
                nn.ReLU(False),
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not batch_norm,
                ),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = nn.Sequential(
            *_create_block(dim=num_features, batch_norm=batch_norm),
            *_create_block(dim=num_features, batch_norm=batch_norm),
        )
        return ResidualBlock(residual)
