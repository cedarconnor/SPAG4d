"""Defines module to compose final Gaussians from base values and delta values.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sharp.models.initializer import GaussianBaseValues
from sharp.utils import math as math_utils
from sharp.utils.color_space import ColorSpace, sRGB2linearRGB
from sharp.utils.gaussians import Gaussians3D

from .params import DeltaFactor


def _get_scale_activation_constant(max_scale: float, min_scale: float) -> tuple[float, float]:
    """Return constants for scale activation function."""
    # To ensure for delta = 0, the value of scale_factor is 1 and the gradient is 1.
    constant_a = (max_scale - min_scale) / (1 - min_scale) / (max_scale - 1)
    constant_b = math_utils.inverse_sigmoid(
        torch.tensor((1.0 - min_scale) / (max_scale - min_scale))
    ).item()
    return constant_a, constant_b


class GaussianComposer(nn.Module):
    """Converts base values and deltas into Gaussians."""

    color_activation_type: math_utils.ActivationType
    opacity_activation_type: math_utils.ActivationType

    def __init__(
        self,
        delta_factor: DeltaFactor,
        min_scale: float,
        max_scale: float,
        color_activation_type: math_utils.ActivationType,
        opacity_activation_type: math_utils.ActivationType,
        color_space: ColorSpace,
        base_scale_on_predicted_mean: bool,
        scale_factor: int = 1,
    ) -> None:
        """Initialize GaussianComposer.

        Args:
            delta_factor: Multiply delta offsets by this factor.
            min_scale: The minimal scale factor for gaussian scale activation.
            max_scale: The maximal scale factor for gaussian scale activation.
            color_activation_type: Which activation function to use for colors.
            opacity_activation_type: Which activation function to use for opacities.
            color_space: Which color space is used in training.
            scale_factor: The scale factor to upsample the delta_values before composition.
            base_scale_on_predicted_mean: Whether to account z offsets for estimating base scale.
        """
        super().__init__()
        self.delta_factor = delta_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.color_activation_type = color_activation_type
        self.opacity_activation_type = opacity_activation_type
        self.color_space = color_space
        self.scale_factor = scale_factor
        self.base_scale_on_predicted_mean = base_scale_on_predicted_mean

    def upsample_delta_value(self, delta: torch.Tensor, scale_factor: int = 1):
        """Upsample the delta value.

        Args:
            delta: The delta values predicted by gaussian predictor.
            scale_factor: The scale factor to upsample the delta_values.
        """
        (
            batch_size,
            num_channels,
            num_layers,
            image_height,
            image_width,
        ) = delta.shape
        new_height = image_height * scale_factor
        new_width = image_width * scale_factor
        upsampled_delta = F.interpolate(
            delta.view(batch_size, num_channels * num_layers, image_height, image_width),
            scale_factor=scale_factor,
        ).view(batch_size, num_channels, num_layers, new_height, new_width)
        return upsampled_delta

    def forward(
        self,
        delta: torch.Tensor,
        base_values: GaussianBaseValues,
        global_scale: torch.Tensor | None = None,
        flatten_output: bool = True,
    ) -> Gaussians3D:
        """Combine predicted delta values with base gaussian values and apply activation function.

        Args:
            delta: The delta values predicted by gaussian predictor.
            base_values: The gaussian base values.
            global_scale: Global scale of Gaussians.
            flatten_output: Flatten the gaussian parameters.

        Returns:
            The computed 3D Gaussians.
        """
        # Upsample the delta if delta and base_values have different strides.
        scale_factor = self.scale_factor
        # For triplane head, the delta has already been upsampled.
        actual_scale_factor = base_values.mean_x_ndc.shape[-1] // delta.shape[-1]
        if scale_factor != 1 and actual_scale_factor != 1:
            delta = self.upsample_delta_value(delta, scale_factor)

        mean_vectors = self._forward_mean(base_values, delta)

        # Account for the change in base scale due to z offsets.
        base_scales = (
            (base_values.scales * base_values.mean_inverse_z_ndc * mean_vectors[:, 2:3, ...])
            if self.base_scale_on_predicted_mean
            else base_values.scales
        )
        singular_values = self._scale_activation(
            base_scales,
            delta[:, 3:6],
            self.min_scale,
            self.max_scale,
        )
        quaternions = self._quaternion_activation(base_values.quaternions, delta[:, 6:10])
        colors = self._color_activation(base_values.colors, delta[:, 10:13])
        opacities = self._opacity_activation(base_values.opacities, delta[:, 13])

        if flatten_output:
            # [B, C, N, H, W] -> [B, N, H, W, C].
            # NOTE: opacities is [B, N, H, W] so it doesn't need to permute.
            mean_vectors = mean_vectors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            singular_values = singular_values.permute(0, 2, 3, 4, 1).flatten(1, 3)
            quaternions = quaternions.permute(0, 2, 3, 4, 1).flatten(1, 3)
            colors = colors.permute(0, 2, 3, 4, 1).flatten(1, 3)
            opacities = opacities.flatten(1, 3)

        # Apply global scaling to convert Gaussians to metric space.
        if global_scale is not None:
            mean_vectors = global_scale[:, None, None] * mean_vectors
            singular_values = global_scale[:, None, None] * singular_values

        return Gaussians3D(
            mean_vectors=mean_vectors,
            singular_values=singular_values,
            quaternions=quaternions,
            colors=colors,
            opacities=opacities,
        )

    def _forward_mean(self, base_values: GaussianBaseValues, delta: torch.Tensor) -> torch.Tensor:
        # Concatenate base vectors and apply mean activation.
        delta_factor = torch.tensor(
            [self.delta_factor.xy, self.delta_factor.xy, self.delta_factor.z],
            device=delta.device,
        )[None, :, None, None, None]

        dtype = base_values.mean_x_ndc.dtype
        device = base_values.mean_x_ndc.device
        target_shape = (1, 3, 1, 1, 1)
        mean_x_mask = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).reshape(
            target_shape
        )
        mean_y_mask = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).reshape(
            target_shape
        )
        mean_z_mask = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).reshape(
            target_shape
        )

        mean_vectors_ndc = (
            base_values.mean_x_ndc.repeat(target_shape) * mean_x_mask
            + base_values.mean_y_ndc.repeat(target_shape) * mean_y_mask
            + base_values.mean_inverse_z_ndc.repeat(target_shape) * mean_z_mask
        )

        mean_vectors = self._mean_activation(mean_vectors_ndc, delta_factor * delta[:, :3])
        return mean_vectors

    def _mean_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        """Mean activation function.

        Args:
            base: Tensor of shape [B, 3, H, W], where first two feature dimensions
                (x,y) are in normalized device coordinates (NDC) where (-1, -1) is
                the top, while the third dimension is inverse depth.
            learned_delta: Tensor of shape [B, 3, H, W] with predicted delta values.

        Returns:
            Returns: The final mean vector after combining base and delta and applying nonlinearies.
        """
        xx = base[:, 0:1] + learned_delta[:, 0:1]
        yy = base[:, 1:2] + learned_delta[:, 1:2]

        a = base[:, 2:3]
        b = learned_delta[:, 2:3]

        # Original formula:
        inverse_zz = F.softplus(math_utils.inverse_softplus(a) + b)
        zz = 1.0 / (inverse_zz + 1e-3)

        mean_vectors = torch.cat([zz * xx, zz * yy, zz], dim=1)
        return mean_vectors

    def _scale_activation(
        self,
        base: torch.Tensor,
        learned_delta: torch.Tensor,
        min_scale: float,
        max_scale: float,
    ) -> torch.Tensor:
        constant_a, constant_b = _get_scale_activation_constant(max_scale, min_scale)
        scale_factor = (max_scale - min_scale) * torch.sigmoid(
            constant_a * self.delta_factor.scale * learned_delta + constant_b
        ) + min_scale
        return base * scale_factor

    def _quaternion_activation(
        self, base: torch.Tensor, learned_delta: torch.Tensor
    ) -> torch.Tensor:
        # No need to normalize the quaternions, since this is also done in rendering.
        return base + self.delta_factor.quaternion * learned_delta

    def _color_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        # For certain activation functions we need to clamp the base value to
        # a supported range.
        if self.color_activation_type == "sigmoid":
            base = torch.clamp(base, min=0.01, max=0.99)
        elif self.color_activation_type in ("exp", "softplus"):
            base = torch.clamp(base, min=0.01)

        activation = math_utils.create_activation_pair(self.color_activation_type)
        colors: torch.Tensor = activation.forward(
            activation.inverse(base) + self.delta_factor.color * learned_delta
        )
        # Convert gaussian color to linear if linearRGB colorspace is specified.
        if self.color_space == "linearRGB":
            colors = sRGB2linearRGB(colors)
        return colors

    def _opacity_activation(self, base: torch.Tensor, learned_delta: torch.Tensor) -> torch.Tensor:
        activation = math_utils.create_activation_pair(self.opacity_activation_type)
        return activation.forward(
            activation.inverse(base) + self.delta_factor.opacity * learned_delta
        )
