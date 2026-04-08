"""Contains different decoders for Gaussian predictor.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from .base_decoder import BaseDecoder
from .monodepth_decoder import (
    create_monodepth_decoder,
)
from .multires_conv_decoder import MultiresConvDecoder, UpsamplingMode
from .unet_decoder import UNetDecoder

__all__ = [
    "BaseDecoder",
    "UNetDecoder",
    "MultiresConvDecoder",
    "UpsamplingMode",
    "create_monodepth_decoder",
]
