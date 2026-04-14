# spag4d/refine/geometric/config.py
from dataclasses import dataclass, field
from typing import Literal

from .consistency import ConsistencyConfig


@dataclass
class ColorPolishConfig:
    steps: int = 500
    lr: float = 1e-3
    sh_degree_max: int = 1
    ssim_weight: float = 0.2
    early_stop_patience: int = 50


@dataclass
class GeometricRefineConfig:
    depth_generator: Literal["da360", "dap"] = "da360"

    # OmniRoam (reuses existing OmniRoamConfig semantics)
    trajectory_mode: str = "auto"
    upscale_backend: Literal["none", "seedvr2"] = "seedvr2"
    max_frames: int = 81

    # Alignment
    align_mode: Literal["scale_shift", "scale_only"] = "scale_shift"
    alpha_confident_threshold: float = 0.90
    depth_gradient_threshold: float = 0.15
    min_inlier_fraction: float = 0.20

    # Hole filtering
    alpha_hole_threshold: float = 0.30
    depth_disoccl_margin_ratio: float = 0.02

    # Cross-frame consistency
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)

    # Aggregation
    voxel_size: float | None = None
    max_new_gaussians: int = 2_000_000

    # Color polish
    color_polish: ColorPolishConfig = field(default_factory=ColorPolishConfig)

    # Diagnostics
    write_diagnostics: bool = True
    save_per_frame_renders: bool = False
