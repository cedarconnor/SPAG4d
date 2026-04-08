"""Tests for spag4d.refine.view_selector.

Covers:
- extract_perspective_crop: shape, dtype, range, yaw sensitivity
- compute_perspective_pose: return type, position, look_at direction
- filter_views_by_gap: threshold, sort order, max-cap, empty input
"""

import math

import numpy as np
import pytest

from spag4d.refine.camera_rig import CameraPose
from spag4d.refine.view_selector import (
    compute_perspective_pose,
    extract_perspective_crop,
    filter_views_by_gap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gradient_panorama(h: int = 64, w: int = 128) -> np.ndarray:
    """Return an ERP panorama whose pixel values vary smoothly across columns."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = np.linspace(0.0, 1.0, w)[None, :]  # R: left-to-right ramp
    img[..., 1] = np.linspace(0.0, 1.0, h)[:, None]  # G: top-to-bottom ramp
    img[..., 2] = 0.5
    return img


def _uniform_panorama(value: float = 0.4, h: int = 64, w: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# TestExtractPerspectiveCrop
# ---------------------------------------------------------------------------

class TestExtractPerspectiveCrop:

    def test_output_shape_default(self):
        erp = _gradient_panorama()
        crop = extract_perspective_crop(erp, yaw_deg=0.0)
        assert crop.shape == (256, 256, 3)

    def test_output_shape_custom_size(self):
        erp = _gradient_panorama()
        crop = extract_perspective_crop(erp, yaw_deg=45.0, size=64)
        assert crop.shape == (64, 64, 3)

    def test_output_dtype_float32(self):
        erp = _gradient_panorama()
        crop = extract_perspective_crop(erp, yaw_deg=0.0, size=32)
        assert crop.dtype == np.float32

    def test_value_range_in_unit_interval(self):
        erp = _gradient_panorama()
        crop = extract_perspective_crop(erp, yaw_deg=30.0, size=64)
        assert crop.min() >= 0.0
        assert crop.max() <= 1.0

    def test_different_yaws_produce_different_crops(self):
        """A gradient panorama should yield distinct crops at different yaw angles."""
        erp = _gradient_panorama(h=128, w=256)
        crop0 = extract_perspective_crop(erp, yaw_deg=0.0, size=64)
        crop90 = extract_perspective_crop(erp, yaw_deg=90.0, size=64)
        assert not np.allclose(crop0, crop90, atol=0.01), (
            "Crops at yaw=0 and yaw=90 should differ on a gradient image"
        )

    def test_opposite_yaws_differ(self):
        erp = _gradient_panorama(h=128, w=256)
        crop0 = extract_perspective_crop(erp, yaw_deg=0.0, size=32)
        crop180 = extract_perspective_crop(erp, yaw_deg=180.0, size=32)
        assert not np.allclose(crop0, crop180, atol=0.05)

    def test_uniform_panorama_produces_uniform_crop(self):
        """A flat-colour ERP must produce a flat-colour crop."""
        value = 0.7
        erp = _uniform_panorama(value=value, h=64, w=128)
        crop = extract_perspective_crop(erp, yaw_deg=0.0, size=32)
        np.testing.assert_allclose(crop, value, atol=0.02)

    def test_pitch_zero_and_nonzero_differ(self):
        """Pitch shift should change which part of the panorama is sampled."""
        erp = _gradient_panorama(h=128, w=256)
        crop_flat = extract_perspective_crop(erp, yaw_deg=0.0, size=32, pitch_deg=0.0)
        crop_up = extract_perspective_crop(erp, yaw_deg=0.0, size=32, pitch_deg=30.0)
        assert not np.allclose(crop_flat, crop_up, atol=0.01)

    def test_fov_affects_content(self):
        """A wider FOV should sample a broader region of the panorama."""
        erp = _gradient_panorama(h=128, w=256)
        crop_narrow = extract_perspective_crop(erp, yaw_deg=0.0, fov_deg=30.0, size=32)
        crop_wide = extract_perspective_crop(erp, yaw_deg=0.0, fov_deg=120.0, size=32)
        # Wider FOV captures more of the gradient, so variance should differ
        assert not np.allclose(crop_narrow, crop_wide, atol=0.01)

    def test_no_nans_or_infs(self):
        erp = _gradient_panorama()
        crop = extract_perspective_crop(erp, yaw_deg=270.0, fov_deg=90.0, size=32)
        assert np.all(np.isfinite(crop))


# ---------------------------------------------------------------------------
# TestComputePerspectivePose
# ---------------------------------------------------------------------------

class TestComputePerspectivePose:

    def test_returns_camera_pose(self):
        pose = compute_perspective_pose(
            translation=np.array([0.0, 0.0, 0.0]),
            yaw_deg=0.0,
        )
        assert isinstance(pose, CameraPose)

    def test_position_matches_translation(self):
        translation = np.array([1.0, 2.0, 3.0])
        pose = compute_perspective_pose(translation=translation, yaw_deg=45.0)
        np.testing.assert_allclose(pose.position, translation)

    def test_look_at_is_ahead_of_position(self):
        translation = np.array([0.0, 0.0, 0.0])
        pose = compute_perspective_pose(translation=translation, yaw_deg=0.0)
        # At yaw=0 the camera looks along +Z, so look_at.Z > position.Z
        assert pose.look_at[2] > pose.position[2] + 0.5

    def test_look_at_yaw_90(self):
        """At yaw=90 deg the forward direction is along +X."""
        translation = np.array([0.0, 0.0, 0.0])
        pose = compute_perspective_pose(translation=translation, yaw_deg=90.0)
        # look_at - position should be approx [1, 0, 0]
        direction = pose.look_at - pose.position
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-6)

    def test_up_vector(self):
        pose = compute_perspective_pose(
            translation=np.array([5.0, 0.0, 5.0]),
            yaw_deg=180.0,
        )
        np.testing.assert_allclose(pose.up, [0.0, 1.0, 0.0])

    def test_fov_propagated(self):
        pose = compute_perspective_pose(
            translation=np.zeros(3),
            yaw_deg=0.0,
            fov_deg=60.0,
        )
        assert pose.fov_deg == 60.0

    def test_size_propagated(self):
        pose = compute_perspective_pose(
            translation=np.zeros(3),
            yaw_deg=0.0,
            size=512,
        )
        assert pose.width == 512
        assert pose.height == 512

    def test_look_at_not_equal_to_position(self):
        pose = compute_perspective_pose(
            translation=np.array([3.0, 1.0, -2.0]),
            yaw_deg=135.0,
        )
        assert not np.allclose(pose.look_at, pose.position)

    def test_different_yaws_produce_different_look_ats(self):
        translation = np.zeros(3)
        pose0 = compute_perspective_pose(translation, yaw_deg=0.0)
        pose90 = compute_perspective_pose(translation, yaw_deg=90.0)
        assert not np.allclose(pose0.look_at, pose90.look_at)


# ---------------------------------------------------------------------------
# TestFilterViewsByGap
# ---------------------------------------------------------------------------

class TestFilterViewsByGap:

    def _make_views(self, ratios):
        return [{"gap_ratio": r, "id": i} for i, r in enumerate(ratios)]

    def test_excludes_below_threshold(self):
        views = self._make_views([0.10, 0.02, 0.20, 0.04])
        result = filter_views_by_gap(views, min_gap_ratio=0.05)
        ratios = [v["gap_ratio"] for v in result]
        assert all(r >= 0.05 for r in ratios)
        assert 0.10 in ratios
        assert 0.20 in ratios
        assert 0.02 not in ratios
        assert 0.04 not in ratios

    def test_sorted_descending(self):
        views = self._make_views([0.10, 0.30, 0.15, 0.25])
        result = filter_views_by_gap(views, min_gap_ratio=0.05)
        ratios = [v["gap_ratio"] for v in result]
        assert ratios == sorted(ratios, reverse=True)

    def test_capped_to_max_views(self):
        views = self._make_views([0.1 * (i + 1) for i in range(20)])
        result = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=5)
        assert len(result) <= 5

    def test_cap_keeps_highest_gap_ratios(self):
        views = self._make_views([0.1 * (i + 1) for i in range(10)])
        result = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=3)
        ratios = [v["gap_ratio"] for v in result]
        # Should keep the three highest ratios
        assert sorted(ratios, reverse=True) == ratios
        assert ratios[0] == pytest.approx(1.0)
        assert ratios[1] == pytest.approx(0.9)
        assert ratios[2] == pytest.approx(0.8)

    def test_empty_input_returns_empty(self):
        result = filter_views_by_gap([], min_gap_ratio=0.05)
        assert result == []

    def test_all_below_threshold_returns_empty(self):
        views = self._make_views([0.01, 0.02, 0.03])
        result = filter_views_by_gap(views, min_gap_ratio=0.05)
        assert result == []

    def test_all_above_threshold_all_included(self):
        views = self._make_views([0.10, 0.20, 0.30])
        result = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=100)
        assert len(result) == 3

    def test_exact_threshold_boundary_included(self):
        """A view with gap_ratio exactly equal to min_gap_ratio is included."""
        views = self._make_views([0.05, 0.10])
        result = filter_views_by_gap(views, min_gap_ratio=0.05)
        ratios = [v["gap_ratio"] for v in result]
        assert 0.05 in ratios

    def test_attribute_access_style(self):
        """Views can expose gap_ratio as an attribute (not just dict key)."""
        class View:
            def __init__(self, ratio):
                self.gap_ratio = ratio

        views = [View(0.10), View(0.01), View(0.25)]
        result = filter_views_by_gap(views, min_gap_ratio=0.05)
        assert len(result) == 2
        assert result[0].gap_ratio == pytest.approx(0.25)
        assert result[1].gap_ratio == pytest.approx(0.10)

    def test_max_views_zero_returns_empty(self):
        views = self._make_views([0.10, 0.20, 0.30])
        result = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=0)
        assert result == []
