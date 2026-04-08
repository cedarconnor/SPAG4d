"""Tests for OmniRoam WSL2 adapter."""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from spag4d.refine.omniroam_adapter import (
    windows_to_wsl_path,
    validate_wsl_environment,
    run_omniroam_wsl,
    extract_video_frames,
)
from spag4d.refine.omniroam_config import OmniRoamConfig


class TestWindowsToWslPath:
    def test_d_drive(self):
        assert windows_to_wsl_path(r"D:\SPAG-4D\output") == "/mnt/d/SPAG-4D/output"

    def test_c_drive(self):
        assert windows_to_wsl_path(r"C:\Users\Cedar\file.jpg") == "/mnt/c/Users/Cedar/file.jpg"

    def test_forward_slashes(self):
        assert windows_to_wsl_path("D:/foo/bar") == "/mnt/d/foo/bar"

    def test_trailing_slash(self):
        result = windows_to_wsl_path(r"D:\SPAG-4D\output\\")
        assert result == "/mnt/d/SPAG-4D/output"


class TestValidateWslEnvironment:
    @patch("spag4d.refine.omniroam_adapter.subprocess.run")
    def test_missing_distro(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        cfg = OmniRoamConfig(wsl_distro="NonExistent")
        with pytest.raises(RuntimeError, match="WSL distro.*not found"):
            validate_wsl_environment(cfg)

    @patch("spag4d.refine.omniroam_adapter.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        cfg = OmniRoamConfig()
        validate_wsl_environment(cfg)  # Should not raise


class TestRunOmniroamWsl:
    @patch("spag4d.refine.omniroam_adapter.subprocess.Popen")
    def test_success(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "Loading model...\n",
            "Step 10/50\n",
            "Step 50/50\n",
            "Saved to output_dir\n",
        ])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        src = tmp_path / "test.jpg"
        src.write_bytes(b"fake jpg")
        out = tmp_path / "out"
        out.mkdir()

        cfg = OmniRoamConfig(enabled=True)
        result = run_omniroam_wsl(str(src), str(out), "forward", cfg)
        assert result == str(out)
        mock_popen.assert_called_once()

    @patch("spag4d.refine.omniroam_adapter.subprocess.Popen")
    def test_failure_raises(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["Error: CUDA OOM\n"])
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        src = tmp_path / "test.jpg"
        src.write_bytes(b"fake jpg")
        out = tmp_path / "out"
        out.mkdir()

        cfg = OmniRoamConfig(enabled=True)
        with pytest.raises(RuntimeError, match="OmniRoam failed"):
            run_omniroam_wsl(str(src), str(out), "forward", cfg)

    @patch("spag4d.refine.omniroam_adapter.subprocess.Popen")
    def test_progress_callback(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["Step 5/50\n", "Step 50/50\n"])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        src = tmp_path / "test.jpg"
        src.write_bytes(b"fake jpg")
        out = tmp_path / "out"
        out.mkdir()

        progress = []
        cfg = OmniRoamConfig(enabled=True)
        run_omniroam_wsl(
            str(src), str(out), "forward", cfg,
            progress_callback=lambda cur, tot: progress.append((cur, tot)),
        )
        assert (5, 50) in progress
        assert (50, 50) in progress


class TestExtractVideoFrames:
    def test_extracts_frames(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10, (8, 4))
        for i in range(3):
            frame = (i * 80 * np.ones((4, 8, 3))).astype(np.uint8)
            writer.write(frame)
        writer.release()

        frames = extract_video_frames(video_path)
        assert len(frames) == 3
        assert frames[0].shape == (4, 8, 3)
        assert frames[0].dtype == np.float32
        assert 0.0 <= frames[0].max() <= 1.0
