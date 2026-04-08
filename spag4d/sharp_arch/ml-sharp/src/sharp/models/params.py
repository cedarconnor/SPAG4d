"""Contains params for backbone.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import dataclasses
from typing import Literal

import sharp.utils.math as math_utils
from sharp.models.blocks import NormLayerName, UpsamplingMode
from sharp.models.presets import ViTPreset
from sharp.utils.color_space import ColorSpace

DimsDecoder = tuple[int, int, int, int, int]
DPTImageEncoderType = Literal["skip_conv", "skip_conv_kernel2"]

ColorInitOption = Literal[
    "none",  # Initialize as gray.
    "first_layer",  # Initialize the first layer with input image, other layers with gray.
    "all_layers",  # Initialize all layers with input image.
]
DepthInitOption = Literal[
    # Initialize the layer of gaussian on surface using min pooling of input depth.
    "surface_min",
    # Initialize the layer of gaussian on surface using max pooling of input depth
    "surface_max",
    # Initialize the layer of gaussian on plane using base_depth depth.
    "base_depth",
    # Initialize the layer of gaussian on plane based on base_depth and index of layer.
    "linear_disparity",
]


@dataclasses.dataclass
class AlignmentParams:
    """Parameters for depth alignment."""

    kernel_size: int = 16
    stride: int = 1
    frozen: bool = False

    # The following parameters are only used for LearnedAlignment.
    # Number of steps in the UNet for LearnedAlignment.
    steps: int = 4
    # Activation type for LearnedAlignment.
    activation_type: math_utils.ActivationType = "exp"
    # Whether to use depth decoder features for LearnedAlignment.
    depth_decoder_features: bool = False
    # Base width of the UNet for LearnedAlignment.
    base_width: int = 16


@dataclasses.dataclass
class DeltaFactor:
    """Factors to multiply deltas with before activation.

    These factors effectively selectively reduce the learning rate.
    """

    xy: float = 0.001
    z: float = 0.001
    color: float = 0.1  # We recommend 0.1 for linearRGB and 1.0 for sRGB.
    opacity: float = 1.0
    scale: float = 1.0
    quaternion: float = 1.0


@dataclasses.dataclass
class InitializerParams:
    """Parameters for initializer."""

    # Common parameters.
    # Multiply scale of Gaussians by this factor.
    scale_factor: float = 1.0
    # Factor to convert inverse depth to disparity.
    disparity_factor: float = 1.0
    # Stride of the initializer.
    stride: int = 2

    # Parameters that only affect MultiLayerInitializer.
    # How many layers of Gaussians to predict (only available for MultiLayerInitializer).
    num_layers: int = 2
    # Which option to use for depth initialization.
    first_layer_depth_option: DepthInitOption = "surface_min"
    rest_layer_depth_option: DepthInitOption = "surface_min"
    # Which option to use for color initialization.
    color_option: ColorInitOption = "all_layers"
    # Which depth value to use for depth layers.
    base_depth: float = 10.0
    # Deactivate gradient for feature inputs.
    feature_input_stop_grad: bool = False
    # Whether to normalize depth to [DepthTransformParam.depth_min,
    # DepthTransformParam.depth_max).
    normalize_depth: bool = True

    # Output only the inpainted layer. In this case, num_layers = 1.
    output_inpainted_layer_only: bool = False
    # Whether to set the uninpainted region to zero opacities.
    set_uninpainted_opacity_to_zero: bool = False
    # Whether to concatenate the inpainting mask to the feature input.
    concat_inpainting_mask: bool = False


@dataclasses.dataclass
class MonodepthParams:
    """Parameters for monodepth network."""

    patch_encoder_preset: ViTPreset = "dinov2l16_384"
    image_encoder_preset: ViTPreset = "dinov2l16_384"

    checkpoint_uri: str | None = None
    unfreeze_patch_encoder: bool = False
    unfreeze_image_encoder: bool = False
    unfreeze_decoder: bool = False
    unfreeze_head: bool = False
    unfreeze_norm_layers: bool = False
    grad_checkpointing: bool = False
    use_patch_overlap: bool = True
    dims_decoder: DimsDecoder = (256, 256, 256, 256, 256)


@dataclasses.dataclass
class MonodepthAdaptorParams:
    """Parameters for monodepth network feature adaptor."""

    encoder_features: bool = True
    decoder_features: bool = False


@dataclasses.dataclass
class GaussianDecoderParams:
    """Parameters for backbone with default values."""

    dim_in: int = 5
    dim_out: int = 32
    # Which normalization to use in backbone.
    norm_type: NormLayerName = "group_norm"
    # How many groups to use for group normalization.
    norm_num_groups: int = 8
    # Stride of backbone.
    stride: int = 2

    patch_encoder_preset: ViTPreset = "dinov2l16_384"
    image_encoder_preset: ViTPreset = "dinov2l16_384"

    # Dimensionality of feature maps for DPT decoder.
    dims_decoder: DimsDecoder = (128, 128, 128, 128, 128)

    # Whether to use depth as input.
    use_depth_input: bool = True

    # Whether to enable gradient checkpointing for the backbone
    grad_checkpointing: bool = False

    # What mode to use for upsampling in decoder.
    upsampling_mode: UpsamplingMode = "transposed_conv"

    # The type of image encoder.
    image_encoder_type: DPTImageEncoderType = "skip_conv_kernel2"


@dataclasses.dataclass
class PredictorParams:
    """Parameters for predictors with default values."""

    # Parameters for submodules.
    initializer: InitializerParams = dataclasses.field(default_factory=InitializerParams)
    monodepth: MonodepthParams = dataclasses.field(default_factory=MonodepthParams)
    monodepth_adaptor: MonodepthAdaptorParams = dataclasses.field(
        default_factory=MonodepthAdaptorParams
    )
    gaussian_decoder: GaussianDecoderParams = dataclasses.field(
        default_factory=GaussianDecoderParams
    )
    # How to align depth map (only relevant for RGBGaussianPredictor).
    depth_alignment: AlignmentParams = dataclasses.field(default_factory=AlignmentParams)

    # Selectively reduce learning rate for different properties.
    delta_factor: DeltaFactor = dataclasses.field(default_factory=DeltaFactor)
    # The maximum scale of Gaussians relative to initial scale.
    max_scale: float = 10.0
    # The minimum scale of Gaussians relative to initial scale.
    min_scale: float = 0.0
    # Which normalization to use in prediction head.
    norm_type: NormLayerName = "group_norm"
    # How many groups to use for group normalization.
    norm_num_groups: int = 8
    # Whether to use predicted mean to sample triplane features.
    use_predicted_mean: bool = False
    # Which activation function to use for colors / opacities.
    color_activation_type: math_utils.ActivationType = "sigmoid"
    opacity_activation_type: math_utils.ActivationType = "sigmoid"
    # Colorspace of the renderer ("linearRGB" or "sRGB").
    color_space: ColorSpace = "linearRGB"
    # A small value to avoid ill-conditioned splats
    low_pass_filter_eps: float = 1e-2
    # How many layer of depth does monodepth model predict.
    num_monodepth_layers: int = 2
    # Whether to sort the monodepth output (for two layer monodepth).
    sorting_monodepth: bool = False
    # Whether to account the z offsets for estimating base scale.
    base_scale_on_predicted_mean: bool = True
