"""Contains preset for monodepth modules.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from .vit import ViTPreset

# Map the decoder configuration with the number of output channels
# for each tensor from the decoder output.
MONODEPTH_ENCODER_DIMS_MAP: dict[ViTPreset, list[int]] = {
    # For publication
    "dinov2l16_384": [256, 512, 1024, 1024],
}

MONODEPTH_HOOK_IDS_MAP: dict[ViTPreset, list[int]] = {
    # For publication
    "dinov2l16_384": [5, 11, 17, 23],
}
