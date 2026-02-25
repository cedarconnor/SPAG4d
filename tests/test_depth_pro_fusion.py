# tests/test_depth_pro_fusion.py
"""
Tests for Phase 1: Cubemap Depth Pro Fusion.

Covers depth_pro_fusion.py without requiring the actual Apple Depth Pro model:
  - _build_face_rays
  - _xyz_to_erp_coords
  - project_erp_to_face / project_erp_to_cubemap
  - align_depth_to_reference
  - composite_faces_to_erp
  - DepthProFusion class (init, not-loaded guard, mock-model integration)
  - ProjectionMode enum
"""

import numpy as np
import pytest


# ── _build_face_rays ──────────────────────────────────────────────────────────

class TestBuildFaceRays:
    def test_shape(self):
        from spag4d.depth_pro_fusion import _build_face_rays
        rays = _build_face_rays("front", face_size=16, fov_deg=90.0)
        assert rays.shape == (16, 16, 3)

    def test_unit_length(self):
        """Every ray must be a unit vector."""
        from spag4d.depth_pro_fusion import _build_face_rays
        for face in ("front", "back", "right", "left", "up", "down"):
            rays = _build_face_rays(face, face_size=16, fov_deg=100.0)
            norms = np.linalg.norm(rays, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                       err_msg=f"Non-unit rays on face {face!r}")

    def test_centre_ray_equals_forward(self):
        """Centre pixel should point exactly in the face-forward direction."""
        from spag4d.depth_pro_fusion import _build_face_rays, _FACE_AXES
        F = 31   # odd so centre pixel is exact
        for face_name, (forward, _, _) in _FACE_AXES.items():
            rays = _build_face_rays(face_name, face_size=F, fov_deg=90.0)
            centre = rays[F // 2, F // 2]
            np.testing.assert_allclose(centre, forward, atol=1e-5,
                                       err_msg=f"Centre ray mismatch on face {face_name!r}")

    def test_dtype(self):
        from spag4d.depth_pro_fusion import _build_face_rays
        rays = _build_face_rays("up", face_size=8, fov_deg=90.0)
        assert rays.dtype == np.float32

    def test_raises_on_face_size_one(self):
        from spag4d.depth_pro_fusion import _build_face_rays
        with pytest.raises(ValueError, match="face_size"):
            _build_face_rays("front", face_size=1, fov_deg=90.0)

    def test_raises_on_unknown_face(self):
        from spag4d.depth_pro_fusion import _build_face_rays
        with pytest.raises(KeyError):
            _build_face_rays("diagonal", face_size=8, fov_deg=90.0)


# ── _xyz_to_erp_coords ────────────────────────────────────────────────────────

class TestXyzToErpCoords:
    def test_north_pole_maps_to_row_zero(self):
        """(0, 1, 0) = north pole → row=0."""
        from spag4d.depth_pro_fusion import _xyz_to_erp_coords
        H, W = 64, 128
        xyz = np.array([[[0., 1., 0.]]])
        rows, _ = _xyz_to_erp_coords(xyz, H, W)
        assert rows[0, 0] < 1.0, f"North pole row={rows[0, 0]:.2f}, expected ≈ 0"

    def test_south_pole_maps_to_row_H(self):
        """(0, -1, 0) = south pole → row≈H."""
        from spag4d.depth_pro_fusion import _xyz_to_erp_coords
        H, W = 64, 128
        xyz = np.array([[[0., -1., 0.]]])
        rows, _ = _xyz_to_erp_coords(xyz, H, W)
        assert rows[0, 0] > H - 1.0, f"South pole row={rows[0, 0]:.2f}, expected ≈ {H}"

    def test_output_within_erp_bounds(self):
        """All ERP coords must fall within the image dimensions."""
        from spag4d.depth_pro_fusion import _build_face_rays, _xyz_to_erp_coords
        H, W = 128, 256
        for face in ("front", "back", "right", "left", "up", "down"):
            rays = _build_face_rays(face, face_size=32, fov_deg=100.0)
            rows, cols = _xyz_to_erp_coords(rays, H, W)
            assert rows.min() >= 0.0 and rows.max() < H + 0.5, \
                f"Row out of bounds on face {face!r}: [{rows.min():.1f}, {rows.max():.1f}]"
            assert cols.min() >= 0.0 and cols.max() < W + 0.5, \
                f"Col out of bounds on face {face!r}: [{cols.min():.1f}, {cols.max():.1f}]"

    def test_equatorial_front_maps_to_correct_col(self):
        """(1, 0, 0) = front of equator should map to col≈0 or col≈W (seam)."""
        from spag4d.depth_pro_fusion import _xyz_to_erp_coords
        H, W = 64, 128
        xyz = np.array([[[1., 0., 0.]]])
        rows, cols = _xyz_to_erp_coords(xyz, H, W)
        # phi = π/2 → row = H/2
        assert abs(rows[0, 0] - H / 2.0) < 1.0, f"Equatorial row={rows[0, 0]:.2f}"
        # theta = 0 → col = W (wraps to 0)
        assert cols[0, 0] < 1.0 or cols[0, 0] > W - 1.0, f"Front col={cols[0, 0]:.2f}"


# ── project_erp_to_face ───────────────────────────────────────────────────────

class TestProjectErpToFace:
    def test_output_shape_rgb(self):
        from spag4d.depth_pro_fusion import project_erp_to_face
        H, W, F = 64, 128, 16
        erp = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        face, uv = project_erp_to_face(erp, "front", face_size=F, fov_deg=90.0)
        assert face.shape == (F, F, 3)
        assert uv.shape == (F, F, 2)

    def test_output_shape_depth(self):
        """Single-channel depth (H, W) should produce (F, F) face."""
        from spag4d.depth_pro_fusion import project_erp_to_face
        H, W, F = 64, 128, 16
        erp = np.ones((H, W), dtype=np.float32) * 5.0
        face, uv = project_erp_to_face(erp, "up", face_size=F, fov_deg=100.0)
        assert face.shape == (F, F)

    def test_uv_cols_within_bounds(self):
        from spag4d.depth_pro_fusion import project_erp_to_face
        H, W, F = 64, 128, 16
        erp = np.zeros((H, W, 3), dtype=np.uint8)
        _, uv = project_erp_to_face(erp, "front", face_size=F, fov_deg=100.0)
        assert uv[..., 0].min() >= 0, "UV col below 0"
        assert uv[..., 0].max() < W + 0.5, "UV col exceeds W"

    def test_uv_rows_within_bounds(self):
        from spag4d.depth_pro_fusion import project_erp_to_face
        H, W, F = 64, 128, 16
        erp = np.zeros((H, W, 3), dtype=np.uint8)
        _, uv = project_erp_to_face(erp, "down", face_size=F, fov_deg=100.0)
        assert uv[..., 1].min() >= 0, "UV row below 0"
        assert uv[..., 1].max() < H + 0.5, "UV row exceeds H"

    def test_raises_on_unknown_face(self):
        from spag4d.depth_pro_fusion import project_erp_to_face
        with pytest.raises(ValueError, match="Unknown face"):
            project_erp_to_face(np.zeros((32, 64, 3), dtype=np.uint8),
                                 "diagonal", face_size=8, fov_deg=90.0)

    def test_uint8_face_stays_uint8(self):
        from spag4d.depth_pro_fusion import project_erp_to_face
        erp = np.full((32, 64, 3), 128, dtype=np.uint8)
        face, _ = project_erp_to_face(erp, "right", face_size=8, fov_deg=90.0)
        assert face.dtype == np.uint8


# ── project_erp_to_cubemap ────────────────────────────────────────────────────

class TestProjectErpToCubemap:
    def test_returns_six_faces(self):
        from spag4d.depth_pro_fusion import project_erp_to_cubemap
        erp = np.zeros((32, 64, 3), dtype=np.uint8)
        faces, uvs = project_erp_to_cubemap(erp, face_size=8, fov_deg=90.0)
        assert set(faces.keys()) == {"front", "back", "right", "left", "up", "down"}
        assert set(uvs.keys())   == {"front", "back", "right", "left", "up", "down"}

    def test_all_face_shapes_correct(self):
        from spag4d.depth_pro_fusion import project_erp_to_cubemap
        erp = np.zeros((64, 128, 3), dtype=np.uint8)
        F = 16
        faces, uvs = project_erp_to_cubemap(erp, face_size=F, fov_deg=100.0)
        for name in faces:
            assert faces[name].shape == (F, F, 3), f"Face {name!r} wrong shape"
            assert uvs[name].shape   == (F, F, 2), f"UV  {name!r} wrong shape"

    def test_sphere_coverage_above_threshold(self):
        """
        With 100° FOV, all 6 faces together should cover >90% of the ERP pixels.

        Note: this test checks *discrete* integer-pixel coverage after UV
        rounding.  The actual cv2.remap does continuous bilinear interpolation
        and achieves 100% coverage; integer rounding leaves a small gap (~7%)
        near the poles where ERP pixels are densely packed, hence the 90%
        threshold rather than 95%.
        """
        from spag4d.depth_pro_fusion import project_erp_to_cubemap
        H, W = 64, 128
        erp = np.zeros((H, W, 3), dtype=np.uint8)
        _, uvs = project_erp_to_cubemap(erp, face_size=128, fov_deg=100.0)

        covered = np.zeros((H, W), dtype=bool)
        for uv in uvs.values():
            cols = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
            rows = np.round(uv[..., 1]).astype(int).clip(0, H - 1)
            covered[rows, cols] = True

        coverage = covered.sum() / covered.size
        assert coverage > 0.90, (
            f"Sphere coverage {coverage:.1%} < 90%.  "
            "Check face axis definitions or FOV."
        )

    def test_adjacent_faces_overlap(self):
        """
        Pairs of adjacent faces must share some ERP pixels (overlap > 0).
        Verifies that the 100° FOV actually creates cross-face overlap.
        """
        from spag4d.depth_pro_fusion import project_erp_to_cubemap
        H, W = 64, 128
        erp = np.zeros((H, W, 3), dtype=np.uint8)
        _, uvs = project_erp_to_cubemap(erp, face_size=32, fov_deg=100.0)

        def pixel_set(uv):
            cols = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
            rows = np.round(uv[..., 1]).astype(int).clip(0, H - 1)
            return set(zip(rows.ravel().tolist(), cols.ravel().tolist()))

        adjacent_pairs = [("front", "right"), ("front", "up"), ("back", "down")]
        for a, b in adjacent_pairs:
            overlap = len(pixel_set(uvs[a]) & pixel_set(uvs[b]))
            assert overlap > 0, f"No overlap between face {a!r} and {b!r}"


# ── align_depth_to_reference ──────────────────────────────────────────────────

class TestAlignDepthToReference:
    def test_identity_scale_unchanged(self):
        """If face_depth already matches dap_depth, scale ≈ 1 and output ≈ input."""
        from spag4d.depth_pro_fusion import align_depth_to_reference
        H, W, F = 64, 128, 16
        dap  = np.full((H, W), 5.0, dtype=np.float32)
        face = np.full((F, F), 5.0, dtype=np.float32)
        # UV map: all pixels point to the centre of the ERP
        uv = np.stack([np.full((F, F), W / 2), np.full((F, F), H / 2)], axis=-1).astype(np.float32)
        aligned = align_depth_to_reference(face, uv, dap)
        np.testing.assert_allclose(aligned.mean(), 5.0, rtol=0.05)

    def test_recovers_known_scale(self):
        """face_depth = dap/k → aligned ≈ dap."""
        from spag4d.depth_pro_fusion import align_depth_to_reference
        H, W, F = 64, 128, 16
        k    = 2.5
        dap  = np.full((H, W), 10.0, dtype=np.float32)
        face = np.full((F, F), 10.0 / k, dtype=np.float32)
        # UV map covering the centre region
        j = np.linspace(W // 4, 3 * W // 4, F, dtype=np.float32)
        i = np.linspace(H // 4, 3 * H // 4, F, dtype=np.float32)
        jj, ii = np.meshgrid(j, i)
        uv = np.stack([jj, ii], axis=-1).astype(np.float32)
        aligned = align_depth_to_reference(face, uv, dap)
        np.testing.assert_allclose(aligned.mean(), 10.0, rtol=0.1)

    def test_output_shape_matches_input(self):
        from spag4d.depth_pro_fusion import align_depth_to_reference
        H, W, F = 32, 64, 8
        dap  = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        face = np.random.uniform(1.0, 10.0, (F, F)).astype(np.float32)
        uv   = np.stack([
            np.random.uniform(0, W - 1, (F, F)),
            np.random.uniform(0, H - 1, (F, F))
        ], axis=-1).astype(np.float32)
        aligned = align_depth_to_reference(face, uv, dap)
        assert aligned.shape == (F, F)

    def test_output_non_negative(self):
        from spag4d.depth_pro_fusion import align_depth_to_reference
        H, W, F = 32, 64, 8
        dap  = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        face = np.random.uniform(1.0, 10.0, (F, F)).astype(np.float32)
        uv   = np.stack([
            np.random.uniform(0, W - 1, (F, F)),
            np.random.uniform(0, H - 1, (F, F))
        ], axis=-1).astype(np.float32)
        aligned = align_depth_to_reference(face, uv, dap)
        assert aligned.min() >= 0.0

    def test_warns_and_returns_unscaled_on_few_valid_pixels(self):
        """When fewer than min_valid_pixels are available, a warning is issued."""
        from spag4d.depth_pro_fusion import align_depth_to_reference
        H, W, F = 4, 4, 4
        dap  = np.zeros((H, W), dtype=np.float32)   # all invalid (≤ 1e-3)
        face = np.full((F, F), 5.0, dtype=np.float32)
        uv   = np.zeros((F, F, 2), dtype=np.float32)
        with pytest.warns(UserWarning):
            aligned = align_depth_to_reference(face, uv, dap, min_valid_pixels=1)
        # Should return unscaled face depth
        np.testing.assert_allclose(aligned, face, atol=1e-5)


# ── composite_faces_to_erp ────────────────────────────────────────────────────

class TestCompositeFacesToErp:
    def _make_face_data(self, H, W, F, depth_val=5.0):
        """Helper: constant-depth face with UV pointing to ERP centre region."""
        from spag4d.depth_pro_fusion import project_erp_to_cubemap
        erp_dummy = np.zeros((H, W, 3), dtype=np.uint8)
        _, uv_maps = project_erp_to_cubemap(erp_dummy, face_size=F, fov_deg=100.0)
        face_depths = {name: np.full((F, F), depth_val, dtype=np.float32)
                       for name in uv_maps}
        return face_depths, uv_maps

    def test_output_shape(self):
        from spag4d.depth_pro_fusion import composite_faces_to_erp
        H, W, F = 32, 64, 8
        face_depths, uv_maps = self._make_face_data(H, W, F)
        depth, conf = composite_faces_to_erp(face_depths, uv_maps, (H, W))
        assert depth.shape == (H, W)
        assert conf.shape  == (H, W)

    def test_depth_non_negative(self):
        from spag4d.depth_pro_fusion import composite_faces_to_erp
        H, W, F = 32, 64, 8
        face_depths, uv_maps = self._make_face_data(H, W, F, depth_val=7.0)
        depth, _ = composite_faces_to_erp(face_depths, uv_maps, (H, W))
        assert depth.min() >= 0.0

    def test_confidence_in_unit_interval(self):
        from spag4d.depth_pro_fusion import composite_faces_to_erp
        H, W, F = 32, 64, 8
        face_depths, uv_maps = self._make_face_data(H, W, F)
        _, conf = composite_faces_to_erp(face_depths, uv_maps, (H, W))
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0 + 1e-6

    def test_gaps_filled_by_dap(self):
        """Pixels with zero coverage weight must be filled by dap_depth."""
        from spag4d.depth_pro_fusion import composite_faces_to_erp
        H, W = 32, 64
        # Use empty face_depths → all ERP pixels are gaps
        depth, conf = composite_faces_to_erp({}, {}, (H, W),
                                              dap_depth=np.full((H, W), 9.0, dtype=np.float32))
        # All covered by DAP
        np.testing.assert_allclose(depth, 9.0, atol=1e-5)
        assert conf.max() == 0.0

    def test_constant_depth_faces_output_near_constant(self):
        """With all faces at the same depth, the composite should be close to that depth."""
        from spag4d.depth_pro_fusion import composite_faces_to_erp
        H, W, F = 64, 128, 32
        face_depths, uv_maps = self._make_face_data(H, W, F, depth_val=4.0)
        dap = np.full((H, W), 4.0, dtype=np.float32)
        depth, _ = composite_faces_to_erp(face_depths, uv_maps, (H, W), dap_depth=dap)
        # Allow some tolerance for edge pixels filled by DAP
        assert abs(depth.mean() - 4.0) < 0.5


# ── DepthProFusion class ───────────────────────────────────────────────────────

class TestDepthProFusion:
    def test_instantiation(self):
        import torch
        from spag4d.depth_pro_fusion import DepthProFusion, ProjectionMode
        fusion = DepthProFusion(torch.device("cpu"), face_size=256)
        assert not fusion.is_loaded
        assert fusion.face_size == 256
        assert fusion.projection_mode == ProjectionMode.CUBEMAP

    def test_fuse_raises_before_load(self):
        """fuse() must raise RuntimeError when load_model() has not been called."""
        import torch
        from spag4d.depth_pro_fusion import DepthProFusion
        fusion = DepthProFusion(torch.device("cpu"), face_size=8)
        with pytest.raises(RuntimeError, match="load_model"):
            fusion.fuse(
                np.zeros((32, 64, 3), dtype=np.uint8),
                np.ones((32, 64), dtype=np.float32),
            )

    def test_load_model_raises_importerror(self):
        """load_model() must raise ImportError when depth_pro is not installed."""
        import sys
        import torch
        from spag4d.depth_pro_fusion import DepthProFusion

        # Temporarily hide depth_pro from imports
        old = sys.modules.get("depth_pro")
        sys.modules["depth_pro"] = None   # type: ignore
        try:
            fusion = DepthProFusion(torch.device("cpu"), face_size=8)
            with pytest.raises(ImportError, match="ml-depth-pro"):
                fusion.load_model()
        finally:
            if old is None:
                sys.modules.pop("depth_pro", None)
            else:
                sys.modules["depth_pro"] = old

    def test_fuse_with_mock_model(self):
        """
        Mock out _model and _transform to test the full fuse() pipeline
        without requiring Apple Depth Pro weights.
        """
        import torch
        from unittest.mock import MagicMock
        from spag4d.depth_pro_fusion import DepthProFusion

        F = 16
        fusion = DepthProFusion(torch.device("cpu"), face_size=F, fov_deg=90.0)

        # Mock model: returns constant depth tensor
        mock_depth = torch.ones(F, F) * 5.0
        mock_model = MagicMock()
        mock_model.infer.return_value = {"depth": mock_depth}
        mock_model.eval.return_value  = mock_model

        mock_transform = MagicMock()
        mock_transform.return_value = torch.zeros(3, F, F)

        fusion._model     = mock_model
        fusion._transform = mock_transform

        H, W = 64, 128
        erp_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        dap_depth = np.full((H, W), 5.0, dtype=np.float32)

        fused_depth, confidence = fusion.fuse(erp_image, dap_depth)

        assert fused_depth.shape == (H, W), "fused_depth shape mismatch"
        assert confidence.shape  == (H, W), "confidence shape mismatch"
        assert fused_depth.min() >= 0.0,     "fused_depth has negative values"
        assert 0.0 <= confidence.max() <= 1.0 + 1e-6, "confidence out of [0,1]"

    def test_icosahedron_mode_warns_and_falls_back(self):
        """ProjectionMode.ICOSAHEDRON should warn and fall back to CUBEMAP."""
        import torch
        from unittest.mock import MagicMock
        from spag4d.depth_pro_fusion import DepthProFusion, ProjectionMode

        F = 8
        fusion = DepthProFusion(torch.device("cpu"), face_size=F,
                                 projection_mode=ProjectionMode.ICOSAHEDRON)

        mock_depth = torch.ones(F, F) * 5.0
        mock_model = MagicMock()
        mock_model.infer.return_value = {"depth": mock_depth}
        mock_model.eval.return_value  = mock_model
        mock_transform = MagicMock()
        mock_transform.return_value = torch.zeros(3, F, F)

        fusion._model     = mock_model
        fusion._transform = mock_transform

        H, W = 32, 64
        with pytest.warns(UserWarning, match="ICOSAHEDRON"):
            fused, conf = fusion.fuse(
                np.zeros((H, W, 3), dtype=np.uint8),
                np.full((H, W), 5.0, dtype=np.float32),
            )
        assert fused.shape == (H, W)


# ── ProjectionMode enum ────────────────────────────────────────────────────────

class TestProjectionModeEnum:
    def test_values(self):
        from spag4d.depth_pro_fusion import ProjectionMode
        assert ProjectionMode.CUBEMAP.value     == "cubemap"
        assert ProjectionMode.ICOSAHEDRON.value == "icosahedron"

    def test_from_string(self):
        from spag4d.depth_pro_fusion import ProjectionMode
        assert ProjectionMode("cubemap")     == ProjectionMode.CUBEMAP
        assert ProjectionMode("icosahedron") == ProjectionMode.ICOSAHEDRON
