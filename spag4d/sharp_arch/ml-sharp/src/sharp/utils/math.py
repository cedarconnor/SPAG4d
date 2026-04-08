"""Contains utility math functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Tuple, Union

import torch
from torch import autograd

ActivationType = Literal[
    "linear",
    "exp",
    "sigmoid",
    "softplus",
    "relu_with_pushback",
    "hard_sigmoid_with_pushback",
]
ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class ActivationPair(NamedTuple):
    """A pair of forward and inverse activation functions."""

    forward: ActivationFunction
    inverse: ActivationFunction


def create_activation_pair(activation_type: ActivationType) -> ActivationPair:
    """Create activation function and corresponding inverse function.

    Args:
        activation_type: The activation type to create.

    Returns:
        The corresponding activation functions and the corresponding inverse function.
    """
    if activation_type == "linear":
        return ActivationPair(lambda x: x, lambda x: x)
    elif activation_type == "exp":
        return ActivationPair(torch.exp, torch.log)
    elif activation_type == "sigmoid":
        return ActivationPair(torch.sigmoid, inverse_sigmoid)
    elif activation_type == "softplus":
        return ActivationPair(torch.nn.functional.softplus, inverse_softplus)
    elif activation_type == "relu_with_pushback":
        return ActivationPair(relu_with_pushback, lambda x: x)
    elif activation_type == "hard_sigmoid_with_pushback":
        return ActivationPair(hard_sigmoid_with_pushback, lambda x: 6.0 * x - 3.0)
    else:
        raise ValueError(f"Unsupported activation function: {activation_type}.")


def inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    """Compute inverse sigmoid."""
    return torch.log(tensor / (1.0 - tensor))


def inverse_softplus(tensor: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    """Compute inverse softplus."""
    tensor = tensor.clamp_min(eps)
    sigmoid = torch.sigmoid(-tensor)
    exp = sigmoid / (1.0 - sigmoid)
    return tensor + torch.log(-exp + 1.0)


# The first value describes the threshold from where clamping will be applied, while
# the second value describes the value to clamp with.
SoftClampRange = Tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]


def softclamp(
    tensor: torch.Tensor,
    min: SoftClampRange | None = None,
    max: SoftClampRange | None = None,
) -> torch.Tensor:
    """Clamp tensor to min/max in differentiable way.

    Args:
        tensor: The tensor to clamp.
        min: Pair of threshold to start clamping and value to clamp to.
            The first value should be larger than the second.
        max: Pair of threshold to start clamping and value to clamp to.
            The first value should be smaller than the second.

    Returns:
        The clamped tensor.
    """

    def normalize(clamp_range: SoftClampRange) -> torch.Tensor:
        value0, value1 = clamp_range
        return value0 + (value1 - value0) * torch.tanh((tensor - value0) / (value1 - value0))

    tensor_clamped = tensor
    if min is not None:
        tensor_clamped = torch.maximum(tensor_clamped, normalize(min))
    if max is not None:
        tensor_clamped = torch.minimum(tensor_clamped, normalize(max))

    return tensor_clamped


class ClampWithPushback(autograd.Function):
    """Implementation of clamp_with_pushback function."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        min: float | None,
        max: float | None,
        pushback: float,
    ) -> torch.Tensor:
        """Apply clamp."""
        if min is not None and max is not None and min >= max:
            raise ValueError("Only min < max is supported.")

        ctx.save_for_backward(tensor)
        ctx.min = min
        ctx.max = max
        ctx.pushback = pushback
        return torch.clamp(tensor, min=min, max=max)

    @staticmethod
    def backward(  # type: ignore[override] # Deal with buggy torch annotations.
        ctx: Any, grad_in: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """Compute gradient of clamp with pushback."""
        grad_out = grad_in.clone()
        (tensor,) = ctx.saved_tensors

        if ctx.min is not None:
            mask_min = tensor < ctx.min
            grad_out[mask_min] = -ctx.pushback

        if ctx.max is not None:
            mask_max = tensor > ctx.max
            grad_out[mask_max] = ctx.pushback

        return grad_out, None, None, None


def clamp_with_pushback(
    tensor: torch.Tensor,
    min: float | None = None,
    max: float | None = None,
    pushback: float = 1e-2,
) -> torch.Tensor:
    """Variant of clamp function which avoid the vanishing gradient problem.

    This function is equivalent to adding a regularizer of the form

        pushback * sum_i (
            relu(min - preactivation_i) + relu(preactivation_i - max)
        )

    to the full loss function, which pushes clamped values back.

    When used in minimization problems, pushback should be greater than
    zero. In maximization problems, pushback should be smaller than zero.
    """
    output = ClampWithPushback.apply(tensor, min, max, pushback)
    assert isinstance(output, torch.Tensor)
    return output


def hard_sigmoid_with_pushback(x: torch.Tensor, slope: float = 1.0 / 6.0) -> torch.Tensor:
    """Apply hard sigmoid with pushback.

    For compatibility reasons, we follow the default PyTorch implementation with a
    default slope of 1/6:

        https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    """
    return clamp_with_pushback(slope * x + 0.5, min=0.0, max=1.0)


def relu_with_pushback(x: torch.Tensor) -> torch.Tensor:
    """Compute relu with pushback."""
    return clamp_with_pushback(x, min=0.0)
