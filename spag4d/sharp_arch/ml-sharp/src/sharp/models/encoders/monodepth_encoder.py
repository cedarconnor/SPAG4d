"""Contains Dense Transformer Prediction architecture.

Implements a variant of Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sharp.models.presets import (
    MONODEPTH_ENCODER_DIMS_MAP,
    MONODEPTH_HOOK_IDS_MAP,
    ViTPreset,
)

from .base_encoder import BaseEncoder
from .spn_encoder import SlidingPyramidNetwork
from .vit_encoder import create_vit


def create_monodepth_encoder(
    patch_encoder_preset: ViTPreset,
    image_encoder_preset: ViTPreset,
    use_patch_overlap: bool = True,
    last_encoder: int = 256,
) -> SlidingPyramidNetwork:
    """Creates DepthDensePredictionTransformer model.

    Args:
        patch_encoder_preset: The preset patch encoder architecture in SPN.
        image_encoder_preset: The preset image encoder architecture in SPN.
        use_patch_overlap: Whether to use overlap between patches in SPN.
        last_encoder: last number of encoder features.
    """
    dims_encoder = [last_encoder] + MONODEPTH_ENCODER_DIMS_MAP[patch_encoder_preset]
    patch_encoder_block_ids = MONODEPTH_HOOK_IDS_MAP[patch_encoder_preset]

    patch_encoder = create_vit(
        preset=patch_encoder_preset,
        intermediate_features_ids=patch_encoder_block_ids,
        # We always need to output intermediate features for assembly.
    )
    image_encoder = create_vit(
        preset=image_encoder_preset,
        intermediate_features_ids=None,
    )

    encoder = SlidingPyramidNetwork(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        use_patch_overlap=use_patch_overlap,
    )

    return encoder


class ProjectionModule(nn.Module):
    """Apply projection of features."""

    def __init__(self, dims_in: list[int], dims_out: list[int]) -> None:
        """Initialize projection module."""
        super().__init__()
        if len(dims_in) != len(dims_out):
            raise ValueError("Length of dims_in must be same as length of dims_out.")
        self.convs = nn.ModuleList(
            [nn.Conv2d(dim_in, dim_out, 1) for dim_in, dim_out in zip(dims_in, dims_out)]
        )

    def forward(self, encodings: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply projection module."""
        if len(encodings) != len(self.convs):
            raise ValueError("Number of encodings must be equal to number of projections.")
        return [conv(encoding) for conv, encoding in zip(self.convs, encodings)]


class MonodepthFeatureEncoder(BaseEncoder):
    """A wrapper around monodepth network to extract features."""

    def __init__(
        self,
        monodepth_encoder: SlidingPyramidNetwork,
        output_dims: list[int] | None = None,
        freeze_projection: bool = False,
    ) -> None:
        """Initialize MonodepthFeatureExtractor."""
        super().__init__()

        self.encoder = monodepth_encoder

        # The monodepth network returns two feature maps for the first entry in
        # backbone.encoder.dims_encoder.
        monodepth_dims = self.encoder.dims_encoder
        monodepth_dims = monodepth_dims

        if output_dims is not None:
            if not len(output_dims) == len(monodepth_dims):
                raise ValueError(
                    "When set, number of output dimensions must be equal to output "
                    f"dimensions of monodepth model {len(monodepth_dims)}."
                )

            self.projection = ProjectionModule(monodepth_dims, output_dims)
            self.output_dims = output_dims
        else:
            self.projection = nn.Identity()
            self.output_dims = monodepth_dims

        if freeze_projection:
            self.projection.requires_grad_(False)

    def forward(self, input_features: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-resolution features."""
        encodings = self.encoder(input_features[:, :3].contiguous())
        return self.projection(encodings)

    def internal_resolution(self) -> int:
        """Internal resolution of the encoder."""
        return self.encoder.internal_resolution()
