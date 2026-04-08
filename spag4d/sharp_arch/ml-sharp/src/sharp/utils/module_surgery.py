"""Contains utility functionality to modify torch modules.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Any

from torch import nn

NORM_LAYER_TYPES = tuple(module_type for name, module_type in nn.__dict__.items() if "Norm" in name)
BATCH_NORM_LAYER_TYPES = tuple(
    module_type for name, module_type in nn.__dict__.items() if "BatchNorm" in name
)


def freeze_norm_layer(module: nn.Module) -> nn.Module:
    """Freeze all normalization layers."""

    def set_module_eval_mode(module: nn.Module, _: Any) -> None:
        module.eval()

    for submodule in module.modules():
        if isinstance(submodule, NORM_LAYER_TYPES):
            submodule.requires_grad_(False)
            # This is to ensure that batch norm layers are always called
            # with the precomputed running statistics.
            submodule.register_forward_pre_hook(set_module_eval_mode)

    return module
