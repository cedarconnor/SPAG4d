"""Contains multi-res convolutional decoder.

Implements the decoder for Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from sharp.models.blocks import FeatureFusionBlock2d, UpsamplingMode
from sharp.utils.training import checkpoint_wrapper

from .base_decoder import BaseDecoder


class MultiresConvDecoder(BaseDecoder):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dims_decoder: Iterable[int] | int,
        grad_checkpointing: bool = False,
        upsampling_mode: UpsamplingMode = "transposed_conv",
    ):
        """Initialize multiresolution convolutional decoder.

        Args:
            dims_encoder: Expected dims at each level from the encoder.
            dims_decoder: Dim of decoder features.
            grad_checkpointing: Whether to checkpoint gradient during training.
            upsampling_mode: What method to use for upsampling.
        """
        super().__init__()
        self.dims_encoder = list(dims_encoder)

        if isinstance(dims_decoder, int):
            self.dims_decoder = [dims_decoder] * len(self.dims_encoder)
        else:
            self.dims_decoder = list(dims_decoder)

        if len(self.dims_decoder) != len(self.dims_encoder):
            raise ValueError("Received dims_encoder and dims_decoder of different sizes.")

        self.dim_out = self.dims_decoder[0]

        num_encoders = len(self.dims_encoder)

        # At the highest resolution, i.e. level 0, we apply projection w/ 1x1 convolution
        # when the dimensions mismatch. Otherwise we do not do anything, which is
        # the default behavior of monodepth.
        conv0 = (
            nn.Conv2d(self.dims_encoder[0], self.dims_decoder[0], kernel_size=1, bias=False)
            if self.dims_encoder[0] != self.dims_decoder[0]
            else nn.Identity()
        )

        convs = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                nn.Conv2d(
                    self.dims_encoder[i],
                    self.dims_decoder[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
        self.convs = nn.ModuleList(convs)

        fusions = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    dim_in=self.dims_decoder[i],
                    dim_out=self.dims_decoder[i - 1] if i != 0 else self.dim_out,
                    upsampling_mode=upsampling_mode if i != 0 else None,
                    batch_norm=False,
                )
            )
        self.fusions = nn.ModuleList(fusions)

        self.grad_checkpointing = grad_checkpointing

    @torch.jit.ignore
    def set_grad_checkpointing(self, is_enabled=True):
        """Enable grad checkpointing."""
        self.grad_checkpointing = is_enabled

    def forward(self, encodings: list[torch.Tensor]) -> torch.Tensor:
        """Decode the multi-resolution encodings."""
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(
                f"Encoder output levels={num_levels} at runtime "
                f"mismatch with expected levels={num_encoders}."
            )

        # Project features of different encoder dims to the same decoder dim.
        # Fuse features from the lowest resolution (num_levels-1)
        # to the highest (0).
        features = self.convs[-1](encodings[-1])
        features = checkpoint_wrapper(self, self.fusions[-1], features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = checkpoint_wrapper(self, self.fusions[i], features, features_i)
        return features
