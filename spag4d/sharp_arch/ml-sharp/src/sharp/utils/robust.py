"""Utility function for loss implementations.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Callable

import torch


def robust_where(
    condition: torch.Tensor,
    input: torch.Tensor,
    branch_true_func: Callable[[torch.Tensor], torch.Tensor],
    branch_false_func: Callable[[torch.Tensor], torch.Tensor],
    branch_true_safe_value: float | None = None,
    branch_false_safe_value: float | None = None,
) -> torch.Tensor:
    """Robust torch.where function to avoid NaN in backward pass.

    See https://github.com/pytorch/pytorch/issues/68425

    Args:
        condition: When True (nonzero), yield branch_true_func(input),
            otherwise yield branch_false_func(input)
        input: The input tensor for torch.where
        branch_true_func: Callable for values at indices where condition is True.
        branch_false_func: Callable for values at indices where condition is False.
        branch_true_safe_value: Safe value to replace the true branch.
        branch_false_safe_value: Safe value to replace the false branch.
    """
    input_1 = input
    input_2 = input
    if branch_true_safe_value is not None:
        input_1 = torch.where(condition, input_1, branch_true_safe_value)
    if branch_false_safe_value is not None:
        input_2 = torch.where(~condition, input_2, branch_false_safe_value)
    return torch.where(
        condition,
        branch_true_func(input_1),
        branch_false_func(input_2),
    )
