"""Tests for spag4d.unisharp360 — wrapper validation and flow."""
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests._unisharp_fixtures import write_fake_unisharp_ply
from spag4d.unisharp360 import convert_unisharp360


def _erp_image(path, w=512, h=256):
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)
    return str(path)


def _make_repo(tmp_path):
    repo = tmp_path / "UniSHARP"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer_unisharp.py").write_text("# stub\n")
    return repo


CPU = torch.device("cpu")


class TestValidation:
    def test_missing_repo_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPAG4D_UNISHARP_REPO", raising=False)
        img = _erp_image(tmp_path / "pano.jpg")
        with pytest.raises(ValueError, match="requires --unisharp-repo"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=None, checkpoint_path="c.pt",
            )

    def test_missing_repo_dir_raises(self, tmp_path):
        img = _erp_image(tmp_path / "pano.jpg")
        with pytest.raises(FileNotFoundError, match="repo not found"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=str(tmp_path / "nope"),
                checkpoint_path="c.pt",
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        repo = _make_repo(tmp_path)
        img = _erp_image(tmp_path / "pano.jpg")
        with pytest.raises(ValueError, match="requires --unisharp-checkpoint"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=str(repo), checkpoint_path=None,
            )

    def test_wrong_aspect_raises(self, tmp_path):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"
        ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "sq.jpg", w=256, h=256)  # 1:1, not 2:1
        with pytest.raises(ValueError, match="2:1 ERP"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            )

    def test_env_fallbacks_used(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"
        ckpt.write_bytes(b"x")
        monkeypatch.setenv("SPAG4D_UNISHARP_REPO", str(repo))
        monkeypatch.setenv("SPAG4D_UNISHARP_CHECKPOINT", str(ckpt))
        img = _erp_image(tmp_path / "sq.jpg", w=256, h=256)  # 1:1 -> fails AFTER env resolve
        # If env fallbacks did NOT resolve, we'd get "requires --unisharp-repo"
        # instead of the aspect error. Asserting the aspect error proves
        # env resolution happened.
        with pytest.raises(ValueError, match="2:1 ERP"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=None, checkpoint_path=None,
            )
