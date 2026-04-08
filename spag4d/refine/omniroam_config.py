"""OmniRoam integration configuration."""

from dataclasses import dataclass, field


@dataclass
class OmniRoamConfig:
    """All settings for OmniRoam-based refinement (Refine v2)."""

    # ── Execution environment ──
    enabled: bool = False
    install_dir: str = "/home/cedarconnor/OmniRoam"  # WSL path
    wsl_distro: str = "Ubuntu"

    # ── Model ──
    ckpt_path: str = "models/OmniRoam/Preview/preview.ckpt"  # relative to install_dir
    base_model_path: str = "models/Wan-AI/Wan2.1-T2V-1.3B"  # relative to install_dir

    # ── Generation ──
    height: int = 480
    width: int = 960
    num_frames: int = 81
    cfg_scale: float = 5.0
    inference_steps: int = 50
    speed: float = 1.0

    # ── Trajectory selection ──
    trajectory_mode: str = "auto"
    available_presets: list = field(default_factory=lambda: [
        "forward", "backward", "left", "right",
    ])
    s_curve_amp_m: float = 1.6
    loop_radius_m: float = 1.5
    step_m: float = 0.25

    # ── View selection ──
    extract_fov_degrees: float = 90.0
    extract_directions: list = field(default_factory=lambda: [0, 90, 180, 270])
    min_gap_ratio: float = 0.05
    max_omniroam_views: int = 200

    # ── Upscale (optional) ──
    upscale_backend: str = "none"  # "none" | "seedvr2"
    seedvr2_model: str = "seedvr2_ema_3b_fp16.safetensors"  # dit_model name (3B=6.4GB, 7B=14GB)
    seedvr2_target_resolution: int = 1024  # short-side pixels (2x from 480)
    seedvr2_color_correction: str = "lab"  # "lab" | "wavelet" | "none"
    seedvr2_block_swap: int = 0  # transformer blocks to swap to CPU (0=disabled)

    # ── Supervision ──
    tier2_weight: float = 0.20
    tier2_warmup_iterations: int = 3000
    hole_mask_threshold: float = 0.3
    hole_mask_update_interval: int = 500

    # ── Gap seeding ──
    gap_seed_stride: int = 4
    gap_seed_initial_opacity: float = 0.01

    # ── Scale alignment ──
    scale_alignment: str = "reprojection"
    scale_search_range: tuple = (0.1, 10.0)
    scale_search_samples: int = 32

    # ── Distillation ──
    iters_per_view: int = 20
    kf_iters: int = 50
    densify_grad_threshold: float = 0.005

    # ── Validation ──
    source_anchor_psnr_floor: float = 25.0
    convergence_threshold: float = 0.02
    max_iterations: int = 3

    # ── Paths for GSFixer baseline comparison ──
    gsfixer_checkpoint: str = "pretrained/gsfix3d"
