"""Contains factory functions to build and load ViT.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging

import timm
import torch

from sharp.models.presets.vit import VIT_CONFIG_DICT, ViTConfig, ViTPreset

LOGGER = logging.getLogger(__name__)


class TimmViT(timm.models.VisionTransformer):
    """Contains TIMM implementation for Vanilla ViT."""

    def __init__(self, config: ViTConfig):
        """Initialize ViT from TIMM implementation."""
        # Handle mlp layers.
        mlp_layer = timm.layers.GluMlp if config.mlp_mode == "glu" else timm.layers.Mlp

        super().__init__(
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            init_values=config.init_values,
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            global_pool=config.global_pool,
            mlp_layer=mlp_layer,
        )

        # Required for extracting intermediate features.
        self.dim_in = config.in_chans
        self.intermediate_features_ids = config.intermediate_features_ids

    def reshape_feature(self, embeddings: torch.Tensor):
        """Discard class token and reshape 1D feature map to a 2D grid."""
        batch_size, seq_len, channel = embeddings.shape

        height, width = self.patch_embed.grid_size

        # Remove class token.
        if self.num_prefix_tokens:
            embeddings = embeddings[:, self.num_prefix_tokens :, :]

        # Shape: (batch, height, width, dim) -> (batch, dim, height, width)
        embeddings = embeddings.reshape(batch_size, height, width, channel).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Override forwarding with intermediate features.

        Adapted from timm ViT.

        Returns:
            Output features and list of features from intermediate layers (patch encoder only).
        """
        intermediate_features = {}

        x = self.patch_embed(input_tensor)
        batch_size, seq_len, _ = x.shape

        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if self.intermediate_features_ids is not None and idx in self.intermediate_features_ids:
                intermediate_features[idx] = x
        x = self.norm(x)

        x = self.reshape_feature(x)
        return x, intermediate_features

    def internal_resolution(self) -> int:
        """Return the internal image size of the network."""
        if isinstance(self.patch_embed.img_size, tuple):
            return self.patch_embed.img_size[0]
        else:
            return self.patch_embed.img_size


def create_vit(
    config: ViTConfig | None = None,
    preset: ViTPreset | None = "dinov2l16_384",
    intermediate_features_ids: list[int] | None = None,
) -> TimmViT:
    """Factory function for creating a ViT model."""
    if config is not None:
        LOGGER.info("Using user-defined config.")
    else:
        if preset is None:
            raise ValueError("User-defined config and preset cannot be both None.")
        LOGGER.info("Using preset ViT %s.", preset)
        config = VIT_CONFIG_DICT[preset]

    config.intermediate_features_ids = intermediate_features_ids
    model = TimmViT(config)
    LOGGER.debug(model)
    return model
