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


from spag4d.unisharp_format import (
    convert_unisharp_ply_to_spag,
    reorient_gaussians_inplace,
    denoise_mask,
)


class TestReorient:
    def test_180_about_x_on_positions(self):
        import numpy as np
        arr = np.empty(1, dtype=[(n, "f4") for n in CORE_VERTEX_FIELDS])
        arr["x"], arr["y"], arr["z"] = 1.0, 2.0, 3.0
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = 1.0, 0.0, 0.0, 0.0
        reorient_gaussians_inplace(arr)
        # (x, y, z) -> (x, -y, -z)
        assert (arr["x"][0], arr["y"][0], arr["z"][0]) == (1.0, -2.0, -3.0)

    def test_identity_quat_maps_to_x180(self):
        import numpy as np
        arr = np.empty(1, dtype=[(n, "f4") for n in CORE_VERTEX_FIELDS])
        # identity rotation (w,x,y,z)=(1,0,0,0) -> (-x, w, -z, y) = (0,1,0,0)
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = 1.0, 0.0, 0.0, 0.0
        reorient_gaussians_inplace(arr)
        q = (arr["rot_0"][0], arr["rot_1"][0], arr["rot_2"][0], arr["rot_3"][0])
        assert q == (0.0, 1.0, 0.0, 0.0)

    def test_reorient_twice_is_identity_rotation(self):
        import numpy as np
        arr = np.empty(3, dtype=[(n, "f4") for n in CORE_VERTEX_FIELDS])
        rng = np.random.RandomState(1)
        for n in CORE_VERTEX_FIELDS:
            arr[n] = rng.rand(3).astype("f4")
        orig = arr.copy()
        reorient_gaussians_inplace(arr)
        reorient_gaussians_inplace(arr)  # 360 deg -> identity rotation
        # Positions and non-rotation fields return exactly.
        for n in ["x", "y", "z", "f_dc_0", "scale_0", "opacity"]:
            np.testing.assert_allclose(arr[n], orig[n], rtol=1e-5)
        # Quaternion returns as -q (same rotation; double cover).
        for n in ["rot_0", "rot_1", "rot_2", "rot_3"]:
            np.testing.assert_allclose(arr[n], -orig[n], rtol=1e-5)


class TestDenoise:
    def test_opacity_threshold_removes_faint(self):
        import numpy as np
        arr = np.zeros(3, dtype=[(n, "f4") for n in CORE_VERTEX_FIELDS])
        # opacity logits: very negative (faint) vs positive (solid)
        arr["opacity"] = np.array([-10.0, 5.0, 5.0], dtype="f4")  # ~0.0, ~0.99, ~0.99
        mask = denoise_mask(arr, opacity_min=0.05, sor_k=0, sor_std_ratio=0.0)
        assert mask.tolist() == [False, True, True]

    def test_sor_removes_isolated_point(self):
        import numpy as np
        n = 60
        arr = np.zeros(n, dtype=[(f, "f4") for f in CORE_VERTEX_FIELDS])
        rng = np.random.RandomState(2)
        pts = rng.rand(n, 3).astype("f4") * 0.1  # tight cluster
        pts[0] = [100.0, 100.0, 100.0]           # one far outlier
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        arr["opacity"] = 5.0
        mask = denoise_mask(arr, opacity_min=0.0, sor_k=8, sor_std_ratio=2.0)
        assert mask[0] == False           # outlier dropped
        assert mask[1:].sum() >= n - 5    # cluster mostly kept


class TestConvert:
    def test_convert_drops_supplements_keeps_core(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=20,
                                      with_supplements=True)
        dst = tmp_path / "out.ply"
        # denoise=False isolates the reorient + clean-rewrite behavior.
        stats = convert_unisharp_ply_to_spag(str(src), str(dst), denoise=False)
        assert stats["num_gaussians"] == 20
        assert stats["converted"] is True

        info = inspect_ply_fields(str(dst))
        assert info["supplement_elements"] == []
        assert info["has_core_fields"] is True

    def test_convert_reorients_vertices(self, tmp_path):
        from plyfile import PlyData
        import numpy as np

        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=12)
        dst = tmp_path / "out.ply"
        convert_unisharp_ply_to_spag(str(src), str(dst), denoise=False)

        a = PlyData.read(str(src))["vertex"]
        b = PlyData.read(str(dst))["vertex"]
        # 180 deg about X: x unchanged, y/z negated.
        np.testing.assert_allclose(np.asarray(b["x"]), np.asarray(a["x"]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(b["y"]), -np.asarray(a["y"]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(b["z"]), -np.asarray(a["z"]), rtol=1e-6)
        # f_dc / scale / opacity untouched
        for f in ["f_dc_0", "scale_0", "opacity"]:
            np.testing.assert_allclose(np.asarray(b[f]), np.asarray(a[f]), rtol=1e-6)

    def test_convert_denoise_reduces_count(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=80)
        dst = tmp_path / "out.ply"
        stats = convert_unisharp_ply_to_spag(str(src), str(dst), denoise=True)
        # denoise should never add gaussians and produces a valid clean PLY
        assert stats["num_gaussians"] <= 80
        assert inspect_ply_fields(str(dst))["supplement_elements"] == []

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
