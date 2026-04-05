"""Tests for SeedVR2 video upscaling adapter."""

import pytest
from unittest.mock import patch, MagicMock

from spag4d.refine.seedvr2_adapter import (
    validate_seedvr2_environment,
    run_seedvr2_upscale,
    _build_seedvr2_args,
)
from spag4d.refine.omniroam_config import OmniRoamConfig


class TestValidateSeedvr2Environment:
    @patch("spag4d.refine.seedvr2_adapter.subprocess.run")
    def test_missing_cli(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        with pytest.raises(RuntimeError, match="SeedVR2.*not found"):
            validate_seedvr2_environment(cfg)

    @patch("spag4d.refine.seedvr2_adapter.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        validate_seedvr2_environment(cfg)  # Should not raise


class TestBuildSeedvr2Args:
    def test_contains_all_required_flags(self):
        cfg = OmniRoamConfig(
            seedvr2_target_resolution=1024,
            seedvr2_batch_size=5,
            seedvr2_color_correction="lab",
            seedvr2_block_swap=36,
            seedvr2_model="seedvr2_ema_7b_sharp_fp16",
        )
        args = _build_seedvr2_args("/mnt/d/video.mp4", "/mnt/d/video_up.mp4", cfg)
        assert "/mnt/d/video.mp4" in args
        assert "--output" in args
        assert "/mnt/d/video_up.mp4" in args
        assert "--resolution" in args
        assert "1024" in args
        assert "--batch_size" in args
        assert "5" in args
        assert "--color_correction" in args
        assert "lab" in args
        assert "--blocks_to_swap" in args
        assert "36" in args
        assert "--dit_model" in args
        assert "--cuda_device" in args

    def test_custom_resolution(self):
        cfg = OmniRoamConfig(seedvr2_target_resolution=2048)
        args = _build_seedvr2_args("/in.mp4", "/out.mp4", cfg)
        idx = args.index("--resolution")
        assert args[idx + 1] == "2048"


class TestRunSeedvr2Upscale:
    @patch("spag4d.refine.seedvr2_adapter.subprocess.Popen")
    def test_success_returns_output_path(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "Loading model...\n",
            "Processing batch 1/5\n",
            "Processing batch 5/5\n",
            "Saved to output\n",
        ])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        input_video = tmp_path / "generated.mp4"
        input_video.write_bytes(b"fake mp4")
        output_video = tmp_path / "generated_upscaled.mp4"

        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        result = run_seedvr2_upscale(str(input_video), str(output_video), cfg)
        assert result == str(output_video)
        mock_popen.assert_called_once()

    @patch("spag4d.refine.seedvr2_adapter.subprocess.Popen")
    def test_failure_raises(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["CUDA OOM\n"])
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        input_video = tmp_path / "generated.mp4"
        input_video.write_bytes(b"fake mp4")

        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        with pytest.raises(RuntimeError, match="SeedVR2 failed"):
            run_seedvr2_upscale(str(input_video), str(tmp_path / "out.mp4"), cfg)

    @patch("spag4d.refine.seedvr2_adapter.subprocess.Popen")
    def test_progress_callback(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "Processing batch 2/5\n",
            "Processing batch 5/5\n",
        ])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        input_video = tmp_path / "generated.mp4"
        input_video.write_bytes(b"fake mp4")
        output_video = tmp_path / "generated_upscaled.mp4"

        progress = []
        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        run_seedvr2_upscale(
            str(input_video), str(output_video), cfg,
            progress_callback=lambda cur, tot: progress.append((cur, tot)),
        )
        assert (2, 5) in progress
        assert (5, 5) in progress
