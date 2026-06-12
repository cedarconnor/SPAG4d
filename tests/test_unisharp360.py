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


class TestFlow:
    def _patch_adapter(self, monkeypatch, tmp_path):
        """Patch run_unisharp_inference to emit a fixture PLY + artifacts."""
        def _fake_run(**kwargs):
            out = Path(kwargs["out_dir"]) / "panos_pano"
            out.mkdir(parents=True, exist_ok=True)
            ply = out / "gaussians.ply"
            write_fake_unisharp_ply(ply, n_vertices=33, with_supplements=True)
            (out / "metadata.json").write_text("{}")
            (out / "forward.gif").write_bytes(b"GIF")
            (out / "rotate.gif").write_bytes(b"GIF")
            return {"returncode": 0, "stdout": "", "stderr": "",
                    "out_dir": kwargs["out_dir"], "ply_path": str(ply),
                    "metadata_path": str(out / "metadata.json")}

        # convert_unisharp360 does `from .unisharp_adapter import
        # run_unisharp_inference` at call time, so patch the source module.
        import spag4d.unisharp_adapter as adapter_mod
        monkeypatch.setattr(adapter_mod, "run_unisharp_inference", _fake_run)

    def test_copy_mode_full_flow(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "pano.jpg")
        self._patch_adapter(monkeypatch, tmp_path)

        out = tmp_path / "out.ply"
        stats = convert_unisharp360(
            input_path=img, output_path=str(out), device=CPU,
            unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            format_mode="copy",
        )
        assert out.exists()
        assert stats["num_gaussians"] == 33
        assert stats["num_faces"] == 0
        assert stats["backend"] == "unisharp"

    def test_save_debug_copies_artifacts(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "pano.jpg")
        self._patch_adapter(monkeypatch, tmp_path)

        out = tmp_path / "out.ply"
        convert_unisharp360(
            input_path=img, output_path=str(out), device=CPU,
            unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            save_debug=True,
        )
        dbg = tmp_path / "out_unisharp_debug"
        assert (dbg / "raw_gaussians.ply").exists()
        assert (dbg / "metadata.json").exists()
        assert (dbg / "forward.gif").exists()

    def test_convert_mode_strips_supplements(self, tmp_path, monkeypatch):
        from spag4d.unisharp_format import inspect_ply_fields
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "pano.jpg")
        self._patch_adapter(monkeypatch, tmp_path)

        out = tmp_path / "out.ply"
        convert_unisharp360(
            input_path=img, output_path=str(out), device=CPU,
            unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            format_mode="convert",
        )
        assert inspect_ply_fields(str(out))["supplement_elements"] == []
