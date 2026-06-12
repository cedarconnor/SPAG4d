"""Tests for the unisharp CLI surface."""
from click.testing import CliRunner

from spag4d.cli import convert


def test_help_lists_unisharp_options():
    runner = CliRunner()
    result = runner.invoke(convert, ["--help"])
    assert result.exit_code == 0
    assert "--sharp-backend" in result.output
    assert "--unisharp-repo" in result.output
    assert "--unisharp-checkpoint" in result.output
    assert "unisharp360" in result.output  # appears in --generator choices help


def test_generator_choice_accepts_unisharp360(tmp_path, monkeypatch):
    """Invoking with --generator unisharp360 reaches converter.convert with
    the unisharp backend args (we stub SPAG4D to avoid real inference)."""
    import numpy as np
    from PIL import Image
    import spag4d.core as core_mod

    img = tmp_path / "pano.jpg"
    Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8)).save(img)
    out = tmp_path / "out.ply"

    captured = {}

    class _StubConverter:
        def __init__(self, *a, **k):
            pass

        def convert(self, **kwargs):
            captured.update(kwargs)

            class _R:
                splat_count = 3
                file_size = 100
                processing_time = 0.0
                depth_range = (0.0, 0.0)
            return _R()

    monkeypatch.setattr(core_mod, "SPAG4D", _StubConverter)

    runner = CliRunner()
    result = runner.invoke(convert, [
        str(img), str(out),
        "--generator", "unisharp360",
        "--unisharp-repo", "/repo",
        "--unisharp-checkpoint", "/ckpt.pt",
    ])
    assert result.exit_code == 0, result.output
    assert captured["unisharp_repo"] == "/repo"
    assert captured["unisharp_checkpoint"] == "/ckpt.pt"
