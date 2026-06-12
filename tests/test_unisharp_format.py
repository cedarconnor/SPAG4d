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


from spag4d.unisharp_format import copy_unisharp_ply_to_output


class TestCopy:
    def test_copy_preserves_vertices_and_supplements(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=17,
                                      with_supplements=True)
        dst = tmp_path / "out.ply"
        stats = copy_unisharp_ply_to_output(str(src), str(dst))
        assert dst.exists()
        assert stats["num_gaussians"] == 17
        assert set(stats["ply_info"]["supplement_elements"]) == {
            "extrinsic", "intrinsic", "image_size"
        }

    def test_copy_is_byte_identical(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=8)
        dst = tmp_path / "out.ply"
        copy_unisharp_ply_to_output(str(src), str(dst))
        assert dst.read_bytes() == src.read_bytes()


from spag4d.unisharp_format import convert_unisharp_ply_to_spag


class TestConvert:
    def test_convert_drops_supplements_keeps_core(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=20,
                                      with_supplements=True)
        dst = tmp_path / "out.ply"
        stats = convert_unisharp_ply_to_spag(str(src), str(dst))
        assert stats["num_gaussians"] == 20
        assert stats["converted"] is True

        info = inspect_ply_fields(str(dst))
        assert info["supplement_elements"] == []
        assert info["has_core_fields"] is True

    def test_convert_preserves_vertex_values(self, tmp_path):
        from plyfile import PlyData
        import numpy as np

        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=12)
        dst = tmp_path / "out.ply"
        convert_unisharp_ply_to_spag(str(src), str(dst))

        a = PlyData.read(str(src))["vertex"]
        b = PlyData.read(str(dst))["vertex"]
        for f in CORE_VERTEX_FIELDS:
            np.testing.assert_allclose(np.asarray(a[f]), np.asarray(b[f]), rtol=1e-6)

    def test_convert_raises_on_missing_core_field(self, tmp_path):
        from plyfile import PlyData, PlyElement
        import numpy as np

        # A PLY missing opacity should raise KeyError.
        bad_fields = [f for f in CORE_VERTEX_FIELDS if f != "opacity"]
        arr = np.zeros(5, dtype=[(n, "f4") for n in bad_fields])
        bad = tmp_path / "bad.ply"
        PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(bad))

        with pytest.raises(KeyError, match="opacity"):
            convert_unisharp_ply_to_spag(str(bad), str(tmp_path / "out.ply"))
