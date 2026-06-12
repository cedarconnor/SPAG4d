"""Tests for spag4d.unisharp_adapter — subprocess command construction."""
import subprocess
import types
from pathlib import Path

import pytest

from tests._unisharp_fixtures import write_fake_unisharp_ply
from spag4d import unisharp_adapter
from spag4d.unisharp_adapter import run_unisharp_inference


def _make_repo(tmp_path):
    """Create a fake UniSHARP repo with the inference script present."""
    repo = tmp_path / "UniSHARP"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer_unisharp.py").write_text("# stub\n")
    return repo


def _fake_run_factory(captured, out_dir, returncode=0, write_ply=True):
    """Return a fake subprocess.run that records args and writes a PLY."""
    def _fake_run(cmd, cwd=None, capture_output=False, text=False, timeout=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        if write_ply and returncode == 0:
            sample = Path(out_dir) / "panos_pano"
            sample.mkdir(parents=True, exist_ok=True)
            write_fake_unisharp_ply(sample / "gaussians.ply")
            (sample / "metadata.json").write_text("{}")
        return types.SimpleNamespace(
            returncode=returncode, stdout="ok-stdout", stderr="err-stderr"
        )
    return _fake_run


class TestCommandConstruction:
    def test_includes_save_ply_and_camera_panorama(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir))

        result = run_unisharp_inference(
            image_path=str(tmp_path / "pano.jpg"),
            out_dir=str(out_dir),
            repo_dir=str(repo),
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            python_exe="python",
        )
        cmd = captured["cmd"]
        assert "--save-ply" in cmd
        assert "--camera" in cmd and cmd[cmd.index("--camera") + 1] == "panorama"
        assert "--checkpoint" in cmd
        assert result["ply_path"].endswith("gaussians.ply")
        assert result["metadata_path"].endswith("metadata.json")

    def test_cwd_is_repo_root(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir))
        run_unisharp_inference(
            image_path=str(tmp_path / "pano.jpg"), out_dir=str(out_dir),
            repo_dir=str(repo), checkpoint_path=str(tmp_path / "ckpt.pt"),
        )
        assert captured["cwd"] == str(repo)


class TestErrorPaths:
    def test_missing_script_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="inference script not found"):
            run_unisharp_inference(
                image_path="x.jpg", out_dir=str(tmp_path / "out"),
                repo_dir=str(tmp_path / "nope"), checkpoint_path="c.pt",
            )

    def test_nonzero_returncode_raises_with_logs(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir, returncode=1,
                                              write_ply=False))
        with pytest.raises(RuntimeError, match="UniSHARP inference failed"):
            run_unisharp_inference(
                image_path="x.jpg", out_dir=str(out_dir),
                repo_dir=str(repo), checkpoint_path="c.pt",
            )

    def test_no_ply_raises(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir, write_ply=False))
        with pytest.raises(FileNotFoundError, match="no gaussians.ply"):
            run_unisharp_inference(
                image_path="x.jpg", out_dir=str(out_dir),
                repo_dir=str(repo), checkpoint_path="c.pt",
            )
