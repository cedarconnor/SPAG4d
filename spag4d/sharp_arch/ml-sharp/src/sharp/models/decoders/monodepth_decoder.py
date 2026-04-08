"""Contains factory function for loading/creating monodepth decoder.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""


from __future__ import annotations

from sharp.models.presets import (
    MONODEPTH_ENCODER_DIMS_MAP,
    ViTPreset,
)

from .multires_conv_decoder import MultiresConvDecoder


def create_monodepth_decoder(
    patch_encoder_preset: ViTPreset,
    dims_decoder=None,
) -> MultiresConvDecoder:
    """Create DepthDensePredictionTransformer model.

    Args:
        patch_encoder_preset: The preset patch encoder architecture in SPN.
        dims_decoder: The decoder architecture.
    """
    dims_encoder = MONODEPTH_ENCODER_DIMS_MAP[patch_encoder_preset]
    if dims_decoder is None:
        dims_decoder = dims_encoder[0]
    if isinstance(dims_decoder, int):
        dims_decoder = [dims_decoder]
    decoder = MultiresConvDecoder(
        dims_encoder=[dims_decoder[0]] + list(dims_encoder), dims_decoder=dims_decoder
    )

    return decoder
