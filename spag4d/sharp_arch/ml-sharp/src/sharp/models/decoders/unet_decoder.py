"""Contains the UNet decoder.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from sharp.models.blocks import (
    NormLayerName,
    norm_layer_2d,
    residual_block_2d,
)

from .base_decoder import BaseDecoder


class UNetDecoder(BaseDecoder):
    """Decoder of UNet model."""

    def __init__(
        self,
        dim_out: int,
        width: List[int] | int,
        steps: int = 5,
        norm_type: NormLayerName = "group_norm",
        norm_num_groups=8,
        blocks_per_layer=2,
    ) -> None:
        """Initialize UNet Decoder.

        Args:
            dim_out: The number of output channels.
            width: Width of last input feature map from encoder
                or the width list of all input feature maps from encoder.
            steps: The number of upsampling steps.
            norm_type: Which kind of normalization layer to use.
            norm_num_groups: How many groups to use for group norm (if relevant).
            blocks_per_layer: How many blocks per layer to use.
        """
        super().__init__()

        if blocks_per_layer < 1:
            raise ValueError("blocks_per_layer must be greater or equal to one.")

        self.dim_out = dim_out

        self.convs_up = nn.ModuleList()

        self.output_dims: list[int]
        # If only one number is specified, we assume each layer will double the channel dimension.
        if isinstance(width, int):
            self.input_dims = [width >> i for i in range(0, steps + 1)]
        else:
            self.input_dims = width[::-1][: steps + 1]

        for i_step in range(steps):
            input_width = self.input_dims[i_step]
            current_width = self.input_dims[i_step + 1]
            convs_up_i = nn.Sequential(
                nn.Upsample(scale_factor=2),
                residual_block_2d(
                    input_width * (1 if i_step == 0 else 2),
                    current_width,
                    norm_type=norm_type,
                    norm_num_groups=norm_num_groups,
                ),
                *[
                    residual_block_2d(
                        current_width,
                        current_width,
                        norm_type=norm_type,
                        norm_num_groups=norm_num_groups,
                    )
                    for _ in range(blocks_per_layer - 1)
                ],
            )
            self.convs_up.append(convs_up_i)
            input_width = 2 * current_width
            current_width //= 2

        last_width = self.input_dims[-1]
        self.conv_out = nn.Sequential(
            norm_layer_2d(last_width * 2, norm_type, num_groups=norm_num_groups),
            nn.ReLU(),
            nn.Conv2d(last_width * 2, dim_out, 1),
            norm_layer_2d(dim_out, norm_type, num_groups=norm_num_groups),
            nn.ReLU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Apply UNet to image.

        Args:
            features: The input multi-level feature map from encoder.

        Returns:
            The output feature map.
        """
        i_feature_layer = len(features) - 1
        out = self.convs_up[0](features[i_feature_layer])
        i_feature_layer -= 1
        for conv_up in self.convs_up[1:]:  # type: ignore
            out = conv_up(torch.cat([out, features[i_feature_layer]], dim=1))
            i_feature_layer -= 1
        out = self.conv_out(torch.cat([out, features[i_feature_layer]], dim=1))

        return out
