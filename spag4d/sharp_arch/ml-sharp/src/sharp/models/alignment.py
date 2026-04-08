"""Contains modules for different types of alignment.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sharp.models.decoders import UNetDecoder
from sharp.models.encoders import UNetEncoder
from sharp.utils import math as math_utils

from .params import AlignmentParams


def create_alignment(
    params: AlignmentParams, depth_decoder_dim: int | None = None
) -> nn.Module | None:
    """Create depth alignment."""
    if depth_decoder_dim is None:
        raise ValueError("Requires depth_decoder_dim for LearnedAlignment.")
    alignment = LearnedAlignment(
        depth_decoder_features=params.depth_decoder_features,
        depth_decoder_dim=depth_decoder_dim,
        steps=params.steps,
        stride=params.stride,
        base_width=params.base_width,
        activation_type=params.activation_type,
    )

    if params.frozen:
        alignment.requires_grad_(False)

    return alignment


class LearnedAlignment(nn.Module):
    """Aligns tensors using a UNet."""

    def __init__(
        self,
        steps: int = 4,
        stride: int = 8,
        base_width: int = 16,
        depth_decoder_features: bool = False,
        depth_decoder_dim: int = 256,
        activation_type: math_utils.ActivationType = "exp",
    ) -> None:
        """Initialize LearnedAlignment.

        Args:
            steps: Number of steps in the UNet.
            stride: Effective downsampling of the alignment module.
            base_width: Base width of the UNet.
            depth_decoder_features: Whether to use depth decoder features.
            depth_decoder_dim: Dimension of the depth decoder features.
            activation_type: Activation type for the alignment output.
        """
        super().__init__()
        self.activation = math_utils.create_activation_pair(activation_type)
        bias_value = self.activation.inverse(torch.tensor(1.0))

        self.depth_decoder_features = depth_decoder_features
        if depth_decoder_features:
            dim_in = 2 + depth_decoder_dim
        else:
            dim_in = 2

        def is_power_of_two(n: int) -> bool:
            """Check if a number is a power of two."""
            if n <= 0:
                return False
            return (n & (n - 1)) == 0

        if not is_power_of_two(stride):
            raise ValueError(f"Stride {stride} is not a power of two.")

        steps_decoder = steps - int(math.log2(stride))
        if steps_decoder < 1:
            raise ValueError(f"{steps_decoder} must be greater or equal to 1.")
        widths = [min(base_width << i, 1024) for i in range(steps + 1)]
        self.encoder = UNetEncoder(dim_in=dim_in, width=widths, steps=steps, norm_num_groups=4)
        self.decoder = UNetDecoder(
            dim_out=widths[0], width=widths, steps=steps_decoder, norm_num_groups=4
        )
        self.conv_out = nn.Conv2d(widths[0], 1, 1, bias=True)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, bias_value)

    def forward(
        self,
        tensor_src: torch.Tensor,
        tensor_tgt: torch.Tensor,
        depth_decoder_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute alignment map."""
        # Since the tensors are usually given by depth which is >= 1.0, we invert
        # the tensors to have them in a reasonable range.
        tensor_src = 1.0 / tensor_src.clamp(min=1e-4)
        tensor_tgt = 1.0 / tensor_tgt.clamp(min=1e-4)
        tensor_input = torch.cat([tensor_src, tensor_tgt], dim=1)
        if self.depth_decoder_features:
            height, width = tensor_src.shape[-2:]
            upsampled_encodings = F.interpolate(
                depth_decoder_features,
                size=(height, width),
                mode="bilinear",
            )
            tensor_input = torch.cat([tensor_input, upsampled_encodings], dim=1)
        features = self.encoder(tensor_input)
        output = self.conv_out(self.decoder(features))
        alignment_map_lowres = self.activation.forward(output)
        if alignment_map_lowres.shape[-2:] != tensor_src.shape[-2]:
            alignment_map = F.interpolate(
                alignment_map_lowres,
                size=tensor_src.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return alignment_map
