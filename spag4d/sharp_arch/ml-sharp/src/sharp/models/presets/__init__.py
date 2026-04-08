"""Contains presets for pretrained neural networks.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from .monodepth import (
    MONODEPTH_ENCODER_DIMS_MAP,
    MONODEPTH_HOOK_IDS_MAP,
)
from .vit import (
    VIT_CONFIG_DICT,
    ViTConfig,
    ViTPreset,
)

__all__ = [
    "ViTConfig",
    "ViTPreset",
    "VIT_CONFIG_DICT",
    "MONODEPTH_ENCODER_DIMS_MAP",
    "MONODEPTH_HOOK_IDS_MAP",
]
