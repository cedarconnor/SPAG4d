"""Contains decoder head for direct prediction of delta values.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
from torch import nn

from .gaussian_decoder import ImageFeatures


class DirectPredictionHead(nn.Module):
    """Decodes features into delta values using convolutions."""

    def __init__(self, feature_dim: int, num_layers: int) -> None:
        """Initialize DirectGaussianPredictor.

        Args:
            feature_dim: Number of input features.
            num_layers: The number of layers of Gaussians to predict.
        """
        super().__init__()
        self.num_layers = num_layers

        # 14 is 3 means, 3 scales, 4 quaternions, 3 colors and 1 opacity
        self.geometry_prediction_head = nn.Conv2d(feature_dim, 3 * num_layers, 1)
        self.geometry_prediction_head.weight.data.zero_()
        assert self.geometry_prediction_head.bias is not None
        self.geometry_prediction_head.bias.data.zero_()

        self.texture_prediction_head = nn.Conv2d(feature_dim, (14 - 3) * num_layers, 1)
        self.texture_prediction_head.weight.data.zero_()
        assert self.texture_prediction_head.bias is not None
        self.texture_prediction_head.bias.data.zero_()

    def forward(self, image_features: ImageFeatures) -> torch.Tensor:
        """Predict deltas for 3D Gaussians.

        Args:
            image_features: Image features from decoder.

        Returns:
            The predicted deltas for Gaussian attributes.
        """
        delta_values_geometry = self.geometry_prediction_head(image_features.geometry_features)
        delta_values_texture = self.texture_prediction_head(image_features.texture_features)
        delta_values_geometry = delta_values_geometry.unflatten(1, (3, self.num_layers))
        delta_values_texture = delta_values_texture.unflatten(1, (14 - 3, self.num_layers))
        delta_values = torch.cat([delta_values_geometry, delta_values_texture], dim=1)
        return delta_values
