"""Configuration for the ArtiFixer3D (Shape A) repair backend.

Mirrors ``omniroam_config.OmniRoamConfig`` in spirit, but holds the WSL/Docker
execution contract validated in the Phase 0/1 evaluation (see
``experiments/artifixer_eval/RESULT.md`` + ``PHASE1.md``).
"""
from dataclasses import dataclass


@dataclass
class ArtiFixer3DConfig:
    """All settings for the ArtiFixer3D Docker/WSL refine backend."""

    # ── Execution environment (validated Phase 0/1) ──
    enabled: bool = False
    wsl_distro: str = "Ubuntu"
    docker_image: str = "artifixer:cuda12"
    artifixer_repo: str = "/home/cedarconnor/ArtiFixer"   # WSL path, mounted to /workspace/artifixer
    checkpoint: str = "/data/artifixer-checkpoints/artifixer-14b.pt"
    wan_mirror: str = "/data/wan_te"                        # local Wan2.1 mirror; --model_id + offline
    torch_cache: str = "/data/torch-cache"                 # persists lib3dgut_cc compile

    # ── Bridge (PLY → COLMAP orbit), matches _bridge_to_colmap.py ──
    num_directions: int = 16
    translation_fracs: tuple = (0.08, 0.20, 0.40)
    render_resolution: int = 640
    fov_deg: float = 70.0
    num_seed_points: int = 80_000

    # ── Anchor/novel split: views with parallax below this quantile are anchors ──
    anchor_parallax_quantile: float = 0.34   # ~lowest third (the 0.08-frac ring)

    # ── ArtiFixer steps ──
    recon_steps: int = 10000                 # prepare 3DGRUT recon
    num_inference_steps: int = 4             # run_inference denoising steps
    distill_steps: int = 30000               # run_artifixer3d 3DGRUT distill
    distill_config: str = "_artifixer_run"   # wrapper config name (non-LPIPS, validated)
    metric_scale: float = 1.0                # cloud is already metric

    # ── Work dir (host Windows path; converted to WSL mount at run time) ──
    work_dir: str = "work/artifixer3d"
