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
