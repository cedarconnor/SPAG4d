"""Tests for the native Windows SeedVR2 adapter (spag4d/seedvr2.py)."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spag4d.seedvr2 import (
    SeedVR2Config,
    build_seedvr2_args,
    validate_seedvr2_environment,
)


# ---------------------------------------------------------------------------
# SeedVR2Config defaults
# ---------------------------------------------------------------------------


class TestSeedVR2ConfigDefaults:
    def test_default_install_dir_is_under_project_root(self):
        cfg = SeedVR2Config()
        # install_dir must be a string ending with seedvr2_videoupscaler
        assert "seedvr2_videoupscaler" in cfg.install_dir.lower()

    def test_default_model(self):
        cfg = SeedVR2Config()
        assert cfg.model == "seedvr2_ema_3b_fp16.safetensors"

    def test_default_target_resolution(self):
        cfg = SeedVR2Config()
        assert cfg.target_resolution == 1024

    def test_default_batch_size(self):
        cfg = SeedVR2Config()
        assert cfg.batch_size == 1

    def test_default_color_correction(self):
        cfg = SeedVR2Config()
        assert cfg.color_correction == "lab"

    def test_default_block_swap(self):
        cfg = SeedVR2Config()
        assert cfg.block_swap == 0

    def test_default_seed(self):
        cfg = SeedVR2Config()
        assert cfg.seed == 42

    def test_default_attention_mode(self):
        cfg = SeedVR2Config()
        assert cfg.attention_mode == "sdpa"


# ---------------------------------------------------------------------------
# build_seedvr2_args
# ---------------------------------------------------------------------------


class TestBuildSeedvr2ArgsImageMode:
    def test_image_mode_has_output_format_png(self):
        cfg = SeedVR2Config()
        args = build_seedvr2_args("/in/face.png", "/out/face.png", cfg, mode="image")
        assert "--output_format" in args
        idx = args.index("--output_format")
        assert args[idx + 1] == "png"

    def test_image_mode_no_video_backend(self):
        cfg = SeedVR2Config()
        args = build_seedvr2_args("/in/face.png", "/out/face.png", cfg, mode="image")
        assert "--video_backend" not in args

    def test_image_mode_common_flags_present(self):
        cfg = SeedVR2Config(
            model="seedvr2_ema_7b_fp16.safetensors",
            target_resolution=2048,
            batch_size=2,
            color_correction="wavelet",
            seed=123,
            attention_mode="flash",
        )
        args = build_seedvr2_args("/in/face.png", "/out/face.png", cfg, mode="image")
        assert "/in/face.png" in args
        assert "--output" in args
        assert "/out/face.png" in args
        assert "--dit_model" in args
        assert "seedvr2_ema_7b_fp16.safetensors" in args
        assert "--resolution" in args
        assert "2048" in args
        assert "--color_correction" in args
        assert "wavelet" in args
        assert "--attention_mode" in args
        assert "flash" in args
        assert "--batch_size" in args
        assert "2" in args
        assert "--seed" in args
        assert "123" in args


class TestBuildSeedvr2ArgsVideoMode:
    def test_video_mode_has_video_backend_opencv(self):
        cfg = SeedVR2Config()
        args = build_seedvr2_args("/in/vid.mp4", "/out/vid.mp4", cfg, mode="video")
        assert "--video_backend" in args
        idx = args.index("--video_backend")
        assert args[idx + 1] == "opencv"

    def test_video_mode_no_output_format(self):
        cfg = SeedVR2Config()
        args = build_seedvr2_args("/in/vid.mp4", "/out/vid.mp4", cfg, mode="video")
        assert "--output_format" not in args

    def test_video_mode_common_flags_present(self):
        cfg = SeedVR2Config(target_resolution=1024)
        args = build_seedvr2_args("/in/vid.mp4", "/out/vid.mp4", cfg, mode="video")
        assert "/in/vid.mp4" in args
        assert "--output" in args
        assert "/out/vid.mp4" in args
        assert "--resolution" in args
        assert "1024" in args


class TestBuildSeedvr2ArgsBlockSwap:
    def test_block_swap_greater_than_zero_adds_offload_flags(self):
        cfg = SeedVR2Config(block_swap=24)
        args = build_seedvr2_args("/in.png", "/out.png", cfg, mode="image")
        assert "--blocks_to_swap" in args
        idx = args.index("--blocks_to_swap")
        assert args[idx + 1] == "24"
        assert "--dit_offload_device" in args
        idx2 = args.index("--dit_offload_device")
        assert args[idx2 + 1] == "cpu"

    def test_block_swap_zero_omits_offload_flags(self):
        cfg = SeedVR2Config(block_swap=0)
        args = build_seedvr2_args("/in.png", "/out.png", cfg, mode="image")
        assert "--blocks_to_swap" not in args
        assert "--dit_offload_device" not in args

    def test_block_swap_36_for_7b_model(self):
        cfg = SeedVR2Config(block_swap=36, model="seedvr2_ema_7b_fp16.safetensors")
        args = build_seedvr2_args("/in.mp4", "/out.mp4", cfg, mode="video")
        assert "--blocks_to_swap" in args
        idx = args.index("--blocks_to_swap")
        assert args[idx + 1] == "36"
        assert "--dit_offload_device" in args


# ---------------------------------------------------------------------------
# validate_seedvr2_environment
# ---------------------------------------------------------------------------


class TestValidateSeedvr2Environment:
    def test_missing_dir_raises_file_not_found(self, tmp_path):
        cfg = SeedVR2Config(install_dir=str(tmp_path / "nonexistent_dir"))
        with pytest.raises(FileNotFoundError, match="inference_cli.py"):
            validate_seedvr2_environment(cfg)

    def test_missing_file_in_existing_dir_raises_file_not_found(self, tmp_path):
        # Directory exists but inference_cli.py is absent
        install_dir = tmp_path / "seedvr2"
        install_dir.mkdir()
        cfg = SeedVR2Config(install_dir=str(install_dir))
        with pytest.raises(FileNotFoundError, match="inference_cli.py"):
            validate_seedvr2_environment(cfg)

    def test_present_dir_and_file_does_not_raise(self, tmp_path):
        install_dir = tmp_path / "seedvr2"
        install_dir.mkdir()
        (install_dir / "inference_cli.py").write_text("# stub")
        cfg = SeedVR2Config(install_dir=str(install_dir))
        # Must not raise
        validate_seedvr2_environment(cfg)
