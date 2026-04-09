"""Tests for spag4d.sharp360 — SHARP 360 pipeline geometry and sampling."""

import math

import numpy as np
import pytest

from spag4d.sharp360 import (
    ExtractionLayout,
    FaceOrientation,
    bilinear_sample,
    bilinear_sample_scalar,
    build_extraction_layout,
    extract_perspective_view,
    make_horizon_view,
)


# ---------------------------------------------------------------------------
# FaceOrientation
# ---------------------------------------------------------------------------

class TestFaceOrientation:
    """Tests for FaceOrientation dataclass."""

    def test_front_rotation_matrix_shape(self):
        """rotation_matrix should be a 3x3 numpy array."""
        view = make_horizon_view(0, 4)  # "front"
        R = view.rotation_matrix
        assert R.shape == (3, 3)

    def test_front_forward_direction(self):
        """Front view (index=0) should face +Z (azimuth 0)."""
        view = make_horizon_view(0, 4)
        # forward should be (0, 0, 1)
        np.testing.assert_allclose(view.forward, [0.0, 0.0, 1.0], atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix columns should be orthonormal."""
        for i in range(6):
            view = make_horizon_view(i, 6)
            R = view.rotation_matrix
            # R^T @ R should be identity
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_determinant(self):
        """Rotation matrix should have |det| = 1 (orthonormal frame).

        The columns are [right, down, forward] which form a camera coordinate
        frame.  When 'down' is world-negative-Y this is a left-handed camera
        frame (det = -1), which is fine — SHARP's apply_transform handles
        reflections via SVD decomposition.
        """
        view = make_horizon_view(0, 6)
        R = view.rotation_matrix
        assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-10

    def test_rotation_matrix_columns(self):
        """Columns of the rotation matrix are [right, down, forward]."""
        view = make_horizon_view(0, 4)  # front view
        R = view.rotation_matrix
        np.testing.assert_allclose(R[:, 0], view.right, atol=1e-10)
        np.testing.assert_allclose(R[:, 1], view.down, atol=1e-10)
        np.testing.assert_allclose(R[:, 2], view.forward, atol=1e-10)


# ---------------------------------------------------------------------------
# make_horizon_view
# ---------------------------------------------------------------------------

class TestMakeHorizonView:
    """Tests for make_horizon_view()."""

    def test_names_4_sides(self):
        """4-side layout should use front/right/back/left."""
        names = [make_horizon_view(i, 4).name for i in range(4)]
        assert names == ["front", "right", "back", "left"]

    def test_names_2_sides(self):
        """2-side layout should use front/back."""
        names = [make_horizon_view(i, 2).name for i in range(2)]
        assert names == ["front", "back"]

    def test_names_6_sides(self):
        """6-side layout should use side_01..side_06."""
        names = [make_horizon_view(i, 6).name for i in range(6)]
        assert names == [f"side_{i + 1:02d}" for i in range(6)]

    def test_right_view_direction(self):
        """Index 1 in a 4-side layout should face +X (azimuth 90 deg)."""
        view = make_horizon_view(1, 4)  # "right"
        np.testing.assert_allclose(view.forward, [1.0, 0.0, 0.0], atol=1e-10)

    def test_back_view_direction(self):
        """Index 2 in a 4-side layout should face -Z (azimuth 180 deg)."""
        view = make_horizon_view(2, 4)  # "back"
        np.testing.assert_allclose(view.forward, [0.0, 0.0, -1.0], atol=1e-10)

    def test_left_view_direction(self):
        """Index 3 in a 4-side layout should face -X (azimuth 270 deg)."""
        view = make_horizon_view(3, 4)  # "left"
        np.testing.assert_allclose(view.forward, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_horizon_views_are_horizontal(self):
        """All horizon views should have Y=0 in their forward direction."""
        for i in range(8):
            view = make_horizon_view(i, 8)
            assert abs(view.forward[1]) < 1e-10, f"View {i}: forward.y = {view.forward[1]}"

    def test_down_vector_matches_sharp_convention(self):
        """All horizon views have down = (0, +1, 0) matching SHARP/Apple convention."""
        for i in range(6):
            view = make_horizon_view(i, 6)
            np.testing.assert_allclose(view.down, [0.0, 1.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# build_extraction_layout
# ---------------------------------------------------------------------------

class TestBuildExtractionLayout:
    """Tests for build_extraction_layout()."""

    def test_view_count(self):
        """Layout should have the requested number of views."""
        layout = build_extraction_layout(512, 1024, side_count=6, overlap_degrees=10.0)
        assert len(layout.views) == 6

    def test_image_dimensions(self):
        """Width is widened for overlap; height = panorama_height."""
        layout = build_extraction_layout(768, 2048, side_count=4, overlap_degrees=10.0)
        # Width should be >= face_size (widened for overlap)
        assert layout.image_width >= 768
        # Height = panorama_height (tall faces for vertical coverage)
        assert layout.image_height == 2048

    def test_focal_length_positive(self):
        """Focal length should be positive and finite."""
        layout = build_extraction_layout(512, 1024, side_count=6, overlap_degrees=10.0)
        assert layout.focal_px > 0
        assert math.isfinite(layout.focal_px)

    def test_focal_length_increases_with_less_overlap(self):
        """Less overlap (narrower FOV) should produce a longer focal length."""
        layout_wide = build_extraction_layout(512, 1024, side_count=6, overlap_degrees=20.0)
        layout_narrow = build_extraction_layout(512, 1024, side_count=6, overlap_degrees=5.0)
        assert layout_narrow.focal_px > layout_wide.focal_px

    def test_focal_x_equals_focal_y(self):
        """Square pixels: focal_x == focal_y."""
        layout = build_extraction_layout(512, 1024, side_count=6)
        assert layout.focal_px == layout.focal_y_px

    def test_layout_name_contains_side_count(self):
        """Layout name should include the side count."""
        layout = build_extraction_layout(512, 1024, side_count=8)
        assert "8" in layout.name


# ---------------------------------------------------------------------------
# extract_perspective_view
# ---------------------------------------------------------------------------

class TestExtractPerspectiveView:
    """Tests for extract_perspective_view()."""

    def test_output_shape(self):
        """Output should match (layout.image_height, layout.image_width, 3)."""
        panorama = np.random.randint(0, 256, (512, 1024, 3), dtype=np.uint8)
        view = make_horizon_view(0, 4)
        layout = build_extraction_layout(256, 512, side_count=4)
        face = extract_perspective_view(
            panorama, layout.image_width, layout.image_height,
            layout.focal_px, layout.focal_y_px, view,
        )
        assert face.shape == (layout.image_height, layout.image_width, 3)

    def test_uniform_panorama_gives_uniform_face(self):
        """A solid-color panorama should produce a solid-color face."""
        color = np.array([100, 150, 200], dtype=np.uint8)
        panorama = np.tile(color, (512, 1024, 1))
        view = make_horizon_view(0, 4)
        layout = build_extraction_layout(64, 512, side_count=4)
        face = extract_perspective_view(
            panorama, layout.image_width, layout.image_height,
            layout.focal_px, layout.focal_y_px, view,
        )
        # Allow +-1 for uint8 rounding
        expected = np.broadcast_to(color, face.shape).astype(float)
        np.testing.assert_allclose(face.astype(float), expected, atol=1.5)

    def test_output_dtype_matches_input(self):
        """Output should be uint8 when input is uint8."""
        panorama = np.zeros((256, 512, 3), dtype=np.uint8)
        view = make_horizon_view(0, 4)
        layout = build_extraction_layout(64, 256, side_count=4)
        face = extract_perspective_view(
            panorama, layout.image_width, layout.image_height,
            layout.focal_px, layout.focal_y_px, view,
        )
        assert face.dtype == np.uint8


# ---------------------------------------------------------------------------
# bilinear_sample (RGB)
# ---------------------------------------------------------------------------

class TestBilinearSample:
    """Tests for bilinear_sample()."""

    def test_integer_coords_exact(self):
        """Sampling at integer coordinates should return exact pixel values."""
        img = np.arange(12, dtype=np.float64).reshape(3, 4, 1).repeat(3, axis=2)
        # Sample at (1.0, 1.0) — should be pixel [1, 1] = 5
        result = bilinear_sample(img, np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(result[0], [5.0, 5.0, 5.0], atol=1e-10)

    def test_horizontal_wrap(self):
        """Sampling beyond the right edge should wrap around horizontally."""
        W = 8
        # Simple gradient: column index as value
        img = np.zeros((1, W, 3), dtype=np.float64)
        for c in range(W):
            img[0, c, :] = float(c)

        # Sample at x = W - 0.5: halfway between last pixel and first (wrapped)
        result = bilinear_sample(img, np.array([W - 0.5]), np.array([0.0]))
        # Should be average of pixel W-1 (=7) and pixel 0 (=0)
        expected = (7.0 + 0.0) / 2.0
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-10)

    def test_batch_shape(self):
        """Output shape should match sample coordinate shape plus channels."""
        img = np.random.rand(10, 20, 3)
        sx = np.random.rand(5, 7) * 19
        sy = np.random.rand(5, 7) * 9
        result = bilinear_sample(img, sx, sy)
        assert result.shape == (5, 7, 3)

    def test_midpoint_interpolation(self):
        """Sampling at (0.5, 0.0) should average pixels [0,0] and [0,1]."""
        img = np.zeros((1, 4, 1), dtype=np.float64)
        img[0, 0, 0] = 10.0
        img[0, 1, 0] = 20.0
        result = bilinear_sample(img, np.array([0.5]), np.array([0.0]))
        np.testing.assert_allclose(result[0, 0], 15.0, atol=1e-10)


# ---------------------------------------------------------------------------
# bilinear_sample_scalar
# ---------------------------------------------------------------------------

class TestBilinearSampleScalar:
    """Tests for bilinear_sample_scalar()."""

    def test_integer_coords_exact(self):
        """Sampling at integer coordinates should return the exact value."""
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = bilinear_sample_scalar(img, np.array([2.0]), np.array([1.0]))
        # pixel [1, 2] = 6
        np.testing.assert_allclose(result[0], 6.0, atol=1e-10)

    def test_horizontal_wrap_scalar(self):
        """Horizontal wrapping should work for scalar maps too."""
        W = 4
        img = np.arange(W, dtype=np.float64).reshape(1, W)  # [0, 1, 2, 3]
        # Sample at x = W - 0.5 = 3.5: average of pixel 3 and pixel 0 (wrapped)
        result = bilinear_sample_scalar(img, np.array([W - 0.5]), np.array([0.0]))
        expected = (3.0 + 0.0) / 2.0
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_batch_shape_scalar(self):
        """Output shape should match sample coordinate shape."""
        img = np.random.rand(10, 20)
        sx = np.random.rand(3, 5) * 19
        sy = np.random.rand(3, 5) * 9
        result = bilinear_sample_scalar(img, sx, sy)
        assert result.shape == (3, 5)
