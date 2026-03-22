"""RefineConfig: all parameters for the refinement pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple


@dataclass
class RefineConfig:
    """Configuration for the SPAG-4D refinement pipeline."""

    # --- Inputs ---
    splat_path: Path = Path("output.ply")
    panorama_rgb_path: Path = Path("panorama.jpg")
    panorama_depth_path: Optional[Path] = None
    output_dir: Path = Path("./refine_output")

    # --- Camera ---
    capture_mode: Literal["keyframe", "path"] = "keyframe"
    camera_preset: str = "orbit"
    camera_vfov_deg: float = 60.0
    render_resolution: Tuple[int, int] = (1920, 1080)
    orbit_radius: float = 0.5
    n_cameras: int = 8

    # --- Panoramic extraction ---
    pano_interp_order: int = 1

    # --- Forward warp ---
    warp_subsample: int = 1
    warp_splatting_method: Literal["nearest", "bilinear"] = "nearest"
    depth_disparity_colormap: str = "turbo"
    depth_disparity_percentile_clip: Tuple[float, float] = (2.0, 98.0)

    # --- Region classification ---
    rgb_error_type_a_threshold: float = 0.08
    depth_disagreement_threshold: float = 0.15
    alpha_gap_threshold: float = 0.3
    type_c_alpha_max: float = 0.1

    # --- Synthesis backend ---
    synthesis_backend: Literal["klein-sharp", "flux-fill", "sdxl", "inspatio"] = "klein-sharp"

    # Klein 9B + ml-sharp LoRA (default)
    klein_base_model: str = "D:/SPAG-4D/models/klein-base-9b"
    klein_sharp_lora: str = "cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair"
    klein_kv_model: str = "black-forest-labs/FLUX.2-klein-9b-kv"
    klein_use_kv_cache: bool = True
    klein_num_steps: int = 30  # base model; distilled model uses 4 steps
    klein_guidance_scale: float = 4.0
    klein_fp8: bool = False

    # FLUX.1 Fill fallback
    flux_fill_model: str = "black-forest-labs/FLUX.1-Fill-dev"
    flux_fill_steps: int = 28
    flux_fill_guidance_scale: float = 3.5

    # SDXL legacy
    sdxl_model: str = "stabilityai/stable-diffusion-xl-inpainting-0.1"
    sdxl_steps: int = 30
    sdxl_guidance_scale: float = 7.5

    # InSpatio-World (Track 2)
    inspatio_checkpoint: str = "./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors"
    inspatio_config: str = "./configs/inference_1.3b.yaml"
    inspatio_wan_model: str = "./checkpoints/Wan2.1-T2V-1.3B"
    inspatio_da3_model: Optional[str] = None
    inspatio_gpu: str = "0"
    inspatio_fps: int = 24

    # --- Depth estimation ---
    depth_model: str = "depth-anything-v2"
    depth_hypotheses: int = 1

    # --- Seeding ---
    shadow_opacity: float = 0.5
    shadow_validation_iters: int = 750
    promotion_consistency_threshold: float = 0.3
    confidence_decay_pixels: int = 150

    # --- Fine-tuning ---
    finetune_iterations: int = 4000
    anchor_loss_weight: float = 8.0
    transition_band_pixels: int = 30
    original_gaussian_lr_scale: float = 0.05

    # --- Validation ---
    max_original_psnr_drop: float = 1.5
    max_refinement_rounds: int = 3

    # --- Hardware ---
    device: str = "cuda"
    mixed_precision: bool = True
