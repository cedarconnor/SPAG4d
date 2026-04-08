"""Contains backbone models for feature extraction from RGBD input.

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

from .base_encoder import BaseEncoder


class UNetEncoder(BaseEncoder):
    """Encoder of UNet model."""

    def __init__(
        self,
        dim_in: int,
        width: List[int] | int,
        steps: int = 6,
        norm_type: NormLayerName = "group_norm",
        norm_num_groups=8,
        blocks_per_layer=2,
    ) -> None:
        """Initialize UNet Encoder.

        Args:
            dim_in: The number of input channels.
            width: Width multiplicator of intermediate layers or the width list of all layers.
            steps: The number of downsampling steps.
            norm_type: Which kind of normalization layer to use.
            norm_num_groups: How many groups to use for group norm (if relevant).
            blocks_per_layer: How many residual blocks per layer to use.
        """
        super().__init__()

        if blocks_per_layer < 1:
            raise ValueError("blocks_per_layer must be greater or equal to one.")

        self.dim_in = dim_in
        self.width = width
        self.num_steps = steps

        self.convs_down = nn.ModuleList()

        self.output_dims: list[int]
        # If only one number is specified, we assume each layer will double the channel dimension.
        if isinstance(width, int):
            self.output_dims = [width << i for i in range(0, steps + 1)]
        else:
            if len(width) != (steps + 1):
                raise ValueError("Length of width should match the steps for UNetEncoder.")
            self.output_dims = width

        self.conv_in = nn.Sequential(
            nn.Conv2d(self.dim_in, self.output_dims[0], 3, stride=1, padding=1),
            norm_layer_2d(self.output_dims[0], norm_type, num_groups=norm_num_groups),
            nn.ReLU(),
        )

        for i_step in range(steps):
            input_width = self.output_dims[i_step]
            current_width = self.output_dims[i_step + 1]
            convs_down_i = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                residual_block_2d(
                    input_width,
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
            self.convs_down.append(convs_down_i)

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Apply UNet Encoder to image.

        Args:
            input: The input image.

        Returns:
            The output multi-level feature map from encoder.
        """
        features = []

        feat_i = self.conv_in(input)
        features.append(feat_i)

        for conv_down in self.convs_down:
            feat_i = conv_down(feat_i)
            features.append(feat_i)

        return features

    @property
    def out_width(self) -> int:
        """Compute the output width for UNet decoder."""
        return self.output_dims[-1]
