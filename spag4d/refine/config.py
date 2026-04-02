"""Refinement pipeline configuration."""

from dataclasses import dataclass, field


@dataclass
class RefineConfig:
    """All hyperparameters for the GSFix3D refinement pipeline."""

    # --- Paths ---
    gsfixer_checkpoint: str = "pretrained/gsfix3d"

    # --- Phase 1: Camera Rig ---
    camera_fov: float = 60.0
    translation_fracs: tuple = (0.05, 0.15, 0.30)
    alpha_threshold: float = 0.1
    num_directions: int = 12
    render_resolution: int = 512

    # --- Phase 1: Camera Selection ---
    min_hole_fraction: float = 0.03
    max_repair_cameras: int = 20

    # --- Phase 2: GSFixer ---
    finetune_steps: int = 500
    finetune_lr: float = 1.0e-5
    inference_steps: int = 50
    guidance_scale: float = 7.5
    mesh_simplify_ratio: float = 0.1

    # --- Phase 3: Distillation ---
    # Values matched to GSFix3D reference (arguments.py OptimizationParams)
    distill_iterations: int = 3000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.005
    lr_position: float = 0.00032
    lr_feature: float = 0.0025
    lr_opacity: float = 0.025
    lr_scaling: float = 0.002
    lr_rotation: float = 0.001
    original_view_ratio: float = 0.3

    # --- Convergence ---
    convergence_threshold: float = 0.02
    max_iterations: int = 3

    # --- Progress callback stages ---
    STAGES: dict = field(default_factory=lambda: {
        "camera_rig": "Generating cameras",
        "mesh_extract": "Extracting mesh",
        "finetune": "Adapting to scene",
        "render_holes": "Detecting holes",
        "gsfixer_inference": "Repairing holes",
        "distill": "Optimizing 3D",
    })
