import pytest
from spag4d.refine.omniroam_config import OmniRoamConfig


def test_defaults():
    cfg = OmniRoamConfig()
    assert cfg.enabled is False
    assert cfg.height == 480
    assert cfg.width == 960
    assert cfg.num_frames == 81
    assert cfg.tier2_weight == 0.20
    assert cfg.upscale_backend == "none"
    assert cfg.scale_alignment == "reprojection"
    assert cfg.trajectory_mode == "auto"
    assert cfg.min_gap_ratio == 0.05
    assert cfg.max_omniroam_views == 200
    assert cfg.gap_seed_stride == 4
    assert cfg.gap_seed_initial_opacity == pytest.approx(0.01)
    assert cfg.extract_fov_degrees == 90.0
    assert cfg.extract_directions == [0, 90, 180, 270]


def test_override():
    cfg = OmniRoamConfig(
        enabled=True,
        tier2_weight=0.30,
        trajectory_mode="all",
        wsl_distro="Ubuntu-22.04",
    )
    assert cfg.enabled is True
    assert cfg.tier2_weight == 0.30
    assert cfg.trajectory_mode == "all"
    assert cfg.wsl_distro == "Ubuntu-22.04"


def test_available_presets_default():
    cfg = OmniRoamConfig()
    assert "forward" in cfg.available_presets
    assert "s_curve" not in cfg.available_presets


def test_seedvr2_defaults():
    cfg = OmniRoamConfig()
    assert cfg.upscale_backend == "none"
    assert cfg.seedvr2_model == "seedvr2_ema_3b_fp16.safetensors"
    assert cfg.seedvr2_target_resolution == 1024
    assert cfg.seedvr2_color_correction == "lab"
    assert cfg.seedvr2_block_swap == 0
    assert not hasattr(cfg, "seedvr2_batch_size")
    assert not hasattr(cfg, "seedvr2_install_dir")


def test_available_presets_independent():
    """Default list should not be shared across instances."""
    a = OmniRoamConfig()
    b = OmniRoamConfig()
    a.available_presets.append("s_curve")
    assert "s_curve" not in b.available_presets
