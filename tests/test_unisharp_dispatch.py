"""Tests for convert_sharp360 backend dispatch + core routing."""
import pytest
import torch

from spag4d.sharp360 import convert_sharp360

CPU = torch.device("cpu")


class TestBackendDispatch:
    def test_unisharp_backend_dispatches(self, monkeypatch):
        called = {}

        def _fake_convert_unisharp360(**kwargs):
            called.update(kwargs)
            return {"num_gaussians": 7, "num_faces": 0,
                    "output_path": kwargs["output_path"],
                    "processing_time": 0.0, "backend": "unisharp"}

        # convert_sharp360 imports convert_unisharp360 lazily from the module.
        import spag4d.unisharp360 as uni
        monkeypatch.setattr(uni, "convert_unisharp360", _fake_convert_unisharp360)

        result = convert_sharp360(
            input_path="pano.jpg", output_path="out.ply", device=CPU,
            backend="unisharp", unisharp_repo="/repo",
            unisharp_checkpoint="/ckpt.pt",
        )
        assert result["num_gaussians"] == 7
        assert called["unisharp_repo"] == "/repo"
        assert called["checkpoint_path"] == "/ckpt.pt"

    def test_hybrid_backend_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="hybrid"):
            convert_sharp360(
                input_path="pano.jpg", output_path="out.ply", device=CPU,
                backend="hybrid",
            )

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown sharp360 backend"):
            convert_sharp360(
                input_path="pano.jpg", output_path="out.ply", device=CPU,
                backend="bogus",
            )


class TestCoreRouting:
    def test_generator_unisharp360_maps_to_backend(self, tmp_path, monkeypatch):
        """SPAG4D.convert(generator='unisharp360') routes to convert_sharp360
        with backend='unisharp'."""
        import numpy as np
        from PIL import Image
        import spag4d.core as core_mod

        # 2:1 ERP so the aspect gate passes.
        img = tmp_path / "pano.jpg"
        Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8)).save(img)
        out = tmp_path / "out.ply"

        captured = {}

        def _fake_convert_sharp360(**kwargs):
            captured.update(kwargs)
            out_p = kwargs["output_path"]
            # Produce a real file so the .stat() in core.py succeeds.
            from tests._unisharp_fixtures import write_fake_unisharp_ply
            write_fake_unisharp_ply(out_p, n_vertices=5)
            return {"num_gaussians": 5, "num_faces": 0, "output_path": out_p}

        # core.py imports convert_sharp360 lazily from spag4d.sharp360.
        import spag4d.sharp360 as s
        monkeypatch.setattr(s, "convert_sharp360", _fake_convert_sharp360)

        conv = core_mod.SPAG4D(device="cpu", generator="unisharp360")
        result = conv.convert(
            input_path=str(img), output_path=str(out),
            generator="unisharp360",
            unisharp_repo="/repo", unisharp_checkpoint="/ckpt.pt",
        )
        assert captured["backend"] == "unisharp"
        assert captured["unisharp_repo"] == "/repo"
        assert result.splat_count == 5
