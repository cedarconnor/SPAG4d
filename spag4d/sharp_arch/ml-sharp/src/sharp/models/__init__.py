"""Contains different Gaussian predictors.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from sharp.models.monodepth import (
    create_monodepth_adaptor,
    create_monodepth_dpt,
)

from .alignment import create_alignment
from .composer import GaussianComposer
from .gaussian_decoder import create_gaussian_decoder
from .heads import DirectPredictionHead
from .initializer import create_initializer
from .params import PredictorParams
from .predictor import RGBGaussianPredictor


def create_predictor(params: PredictorParams) -> RGBGaussianPredictor:
    """Create gaussian predictor model specified by name."""
    if params.gaussian_decoder.stride < params.initializer.stride:
        raise ValueError(
            "We donot expected gaussian_decoder has higher resolution than initializer."
        )

    scale_factor = params.gaussian_decoder.stride // params.initializer.stride
    gaussian_composer = GaussianComposer(
        delta_factor=params.delta_factor,
        min_scale=params.min_scale,
        max_scale=params.max_scale,
        color_activation_type=params.color_activation_type,
        opacity_activation_type=params.opacity_activation_type,
        color_space=params.color_space,
        scale_factor=scale_factor,
        base_scale_on_predicted_mean=params.base_scale_on_predicted_mean,
    )
    if params.num_monodepth_layers > 1 and params.initializer.num_layers != 2:
        raise KeyError("We only support num_layers = 2 when num_monodepth_layers > 1.")

    monodepth_model = create_monodepth_dpt(params.monodepth)
    monodepth_adaptor = create_monodepth_adaptor(
        monodepth_model,
        params.monodepth_adaptor,
        params.num_monodepth_layers,
        params.sorting_monodepth,
    )

    if params.num_monodepth_layers == 2:
        monodepth_adaptor.replicate_head(params.num_monodepth_layers)

    gaussian_decoder = create_gaussian_decoder(
        params.gaussian_decoder,
        dims_depth_features=monodepth_adaptor.get_feature_dims(),
    )
    initializer = create_initializer(
        params.initializer,
    )
    prediction_head = DirectPredictionHead(
        feature_dim=gaussian_decoder.dim_out, num_layers=initializer.num_layers
    )
    decoder_dim = monodepth_model.decoder.dims_decoder[-1]
    return RGBGaussianPredictor(
        init_model=initializer,
        feature_model=gaussian_decoder,
        prediction_head=prediction_head,
        monodepth_model=monodepth_adaptor,
        gaussian_composer=gaussian_composer,
        scale_map_estimator=create_alignment(params.depth_alignment, depth_decoder_dim=decoder_dim),
    )


__all__ = [
    "PredictorParams",
    "create_predictor",
]
