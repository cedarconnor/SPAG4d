"""Contains Dense Transformer Prediction architecture.

Implements a variant of Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from sharp.models.blocks import (
    FeatureFusionBlock2d,
    NormLayerName,
    residual_block_2d,
)
from sharp.models.decoders import BaseDecoder, MultiresConvDecoder
from sharp.models.params import DPTImageEncoderType, GaussianDecoderParams


def create_gaussian_decoder(
    params: GaussianDecoderParams, dims_depth_features: list[int]
) -> GaussianDensePredictionTransformer:
    """Create gaussian_decoder model specified by gaussian_decoder_name."""
    decoder = MultiresConvDecoder(
        dims_depth_features,
        params.dims_decoder,
        grad_checkpointing=params.grad_checkpointing,
        upsampling_mode=params.upsampling_mode,
    )

    return GaussianDensePredictionTransformer(
        decoder=decoder,
        dim_in=params.dim_in,
        dim_out=params.dim_out,
        stride_out=params.stride,
        norm_type=params.norm_type,
        norm_num_groups=params.norm_num_groups,
        use_depth_input=params.use_depth_input,
        grad_checkpointing=params.grad_checkpointing,
        image_encoder_type=params.image_encoder_type,
        image_encoder_params=params,
    )


def _create_project_upsample_block(
    dim_in: int,
    dim_out: int,
    upsample_layers: int,
    dim_intermediate: int | None = None,
) -> nn.Module:
    if dim_intermediate is None:
        dim_intermediate = dim_out
    # Projection.
    blocks = [
        nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_intermediate,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
    ]

    # Upsampling.
    blocks += [
        nn.ConvTranspose2d(
            in_channels=dim_intermediate if i == 0 else dim_out,
            out_channels=dim_out,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        for i in range(upsample_layers)
    ]

    return nn.Sequential(*blocks)


class ImageFeatures(NamedTuple):
    """Image feature extracted from decoder."""

    texture_features: torch.Tensor
    geometry_features: torch.Tensor


class SkipConvBackbone(nn.Module):
    """A wrapper around a conv layer that behaves like a BaseBackbone."""

    def __init__(self, dim_in: int, dim_out: int, kernel_size: int, stride_out: int):
        """Initialize SkipConvBackbone."""
        super().__init__()
        self.stride_out = stride_out
        if stride_out == 1 and kernel_size != 1:
            raise ValueError("We only support kernel_size = 1 if stride_out is 1.")
        padding: int = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, stride=stride_out, padding=padding
        )

    def forward(
        self,
        input_features: torch.Tensor,
        encodings: list[torch.Tensor] | None = None,
    ) -> ImageFeatures:
        """Apply SkipConvBackbone to image."""
        output = self.conv(input_features)
        return ImageFeatures(
            texture_features=output,
            geometry_features=output,
        )

    @property
    def stride(self) -> int:
        """Effective downsampling stride."""
        return self.stride_out


class GaussianDensePredictionTransformer(nn.Module):
    """Dense Prediction Transformer for Gaussian.

    Reuse monodepth decoded features for processing.
    """

    norm_type: NormLayerName

    def __init__(
        self,
        decoder: BaseDecoder,
        dim_in: int,
        dim_out: int,
        stride_out: int,
        image_encoder_params: GaussianDecoderParams,
        image_encoder_type: DPTImageEncoderType = "skip_conv",
        norm_type: NormLayerName = "group_norm",
        norm_num_groups: int = 8,
        use_depth_input: bool = True,
        grad_checkpointing: bool = False,
    ):
        """Initialize Dense Prediction Transformer for Gaussian.

        Args:
            decoder: Decoder to decode features.
            monodepth_decoder: Optional monodepth decoder to fuse monodepth decoded features.
            dim_in: Input dimension.
            dim_out: Final output dimension.
            stride_out: Stride of output feature map.
            image_encoder_params: The backbone parameters to configurate the image encoder.
            image_encoder_type: Type of image encoder to use.
            encoder: Encoder to generate features using monodepth model.
            norm_type: Type of norm layers.
            norm_num_groups: Num groups for norm layers.
            use_depth_input: Whether to use depth input.
            grad_checkpointing: Whether to use gradient checkpointing.
        """
        super().__init__()

        self.decoder = decoder
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride_out = stride_out
        self.norm_type = norm_type
        self.norm_num_groups = norm_num_groups
        self.use_depth_input = use_depth_input
        self.grad_checkpointing = grad_checkpointing
        self.image_encoder_type = image_encoder_type

        # Adopt an image encoder to lift dimension to monodepth feature and
        # resize to be the same resolution as the decoder output.
        dim_in = self.dim_in if use_depth_input else self.dim_in - 1
        image_encoder_params.dim_in = dim_in
        image_encoder_params.dim_out = decoder.dim_out
        self.image_encoder = self._create_image_encoder(image_encoder_params, stride_out)

        self.fusion = FeatureFusionBlock2d(decoder.dim_out)

        if stride_out == 1:
            self.upsample = _create_project_upsample_block(
                decoder.dim_out,
                decoder.dim_out,
                upsample_layers=1,
            )
        elif stride_out == 2:
            self.upsample = nn.Identity()
        else:
            raise ValueError("We only support stride is 1 or 2 for DPT backbone.")

        self.texture_head = self._create_head(dim_decoder=decoder.dim_out, dim_out=self.dim_out)
        self.geometry_head = self._create_head(dim_decoder=decoder.dim_out, dim_out=self.dim_out)

    def _create_head(self, dim_decoder: int, dim_out: int) -> nn.Module:
        return nn.Sequential(
            residual_block_2d(
                dim_in=dim_decoder,
                dim_out=dim_decoder,
                dim_hidden=dim_decoder // 2,
                norm_type=self.norm_type,
                norm_num_groups=self.norm_num_groups,
            ),
            residual_block_2d(
                dim_in=dim_decoder,
                dim_hidden=dim_decoder // 2,
                dim_out=dim_decoder,
                norm_type=self.norm_type,
                norm_num_groups=self.norm_num_groups,
            ),
            nn.ReLU(),
            nn.Conv2d(dim_decoder, dim_out, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def _create_image_encoder(
        self, image_encoder_params: GaussianDecoderParams, stride_out: int
    ) -> nn.Module:
        """Create image encoder and return based on parameters."""
        if self.image_encoder_type == "skip_conv":
            # Use kernel_size = 1 only if stride_out is 1.
            return SkipConvBackbone(
                image_encoder_params.dim_in,
                image_encoder_params.dim_out,
                kernel_size=3 if stride_out != 1 else 1,
                stride_out=stride_out,
            )
        elif self.image_encoder_type == "skip_conv_kernel2":
            return SkipConvBackbone(
                image_encoder_params.dim_in,
                image_encoder_params.dim_out,
                kernel_size=stride_out,
                stride_out=stride_out,
            )
        else:
            raise ValueError(f"Unsupported image encoder type: {self.image_encoder_type}")

    def forward(self, input_features: torch.Tensor, encodings: list[torch.Tensor]) -> ImageFeatures:
        """Run monodepth and fuse features with input image to predict Gaussians.

        Args:
            input_features: The input features to use.
            encodings: Feature encodings (e.g. from monodepth network).
        """
        features = self.decoder(encodings).contiguous()
        features = self.upsample(features)

        if self.use_depth_input:
            skip_features = self.image_encoder(input_features).texture_features
        else:
            skip_features = self.image_encoder(input_features[:, :3].contiguous())
        features = self.fusion(features, skip_features)

        texture_features = self.texture_head(features)
        geometry_features = self.geometry_head(features)

        return ImageFeatures(
            texture_features=texture_features,  # type: ignore
            geometry_features=geometry_features,  # type: ignore
        )

    @property
    def stride(self) -> int:
        """Internal stride of GaussianDensePredictionTransformer."""
        return self.stride_out
