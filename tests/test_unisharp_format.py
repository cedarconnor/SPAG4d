"""Tests for spag4d.unisharp_format — PLY inspect/count/copy/convert."""
from pathlib import Path

import pytest

from tests._unisharp_fixtures import CORE_VERTEX_FIELDS, write_fake_unisharp_ply
from spag4d.unisharp_format import (
    count_ply_vertices,
    inspect_ply_fields,
)


class TestInspect:
    def test_detects_core_fields(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", n_vertices=10)
        info = inspect_ply_fields(str(ply))
        assert info["has_core_fields"] is True
        for f in CORE_VERTEX_FIELDS:
            assert f in info["vertex_fields"]

    def test_detects_supplement_elements(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", with_supplements=True)
        info = inspect_ply_fields(str(ply))
        assert set(info["supplement_elements"]) == {"extrinsic", "intrinsic", "image_size"}

    def test_no_supplements_when_absent(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", with_supplements=False)
        info = inspect_ply_fields(str(ply))
        assert info["supplement_elements"] == []

    def test_reads_color_space_comment(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", color_space="linearRGB")
        info = inspect_ply_fields(str(ply))
        assert info["color_space"] == "linearRGB"

    def test_color_space_none_when_absent(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", color_space=None)
        info = inspect_ply_fields(str(ply))
        assert info["color_space"] is None


class TestCount:
    def test_counts_vertices(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", n_vertices=42)
        assert count_ply_vertices(str(ply)) == 42
