"""Tests for spag4d.refine.gap_seeding."""

import numpy as np
import pytest

from spag4d.refine.gap_seeding import compute_erp_ray_directions, seed_gap_gaussians


class TestComputeErpRayDirections:
    """Tests for compute_erp_ray_directions."""

    def test_output_shape(self):
        rays = compute_erp_ray_directions(h=32, w=64)
        assert rays.shape == (32, 64, 3)

    def test_output_dtype_float32(self):
        rays = compute_erp_ray_directions(h=16, w=32)
        assert rays.dtype == np.float32

    def test_unit_vectors(self):
        rays = compute_erp_ray_directions(h=32, w=64)
        norms = np.linalg.norm(rays, axis=-1)
        # All rays should be unit length — allow a small tolerance for float32.
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_unit_vectors_small_image(self):
        """Even a 2x4 image should produce unit vectors."""
        rays = compute_erp_ray_directions(h=2, w=4)
        norms = np.linalg.norm(rays, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_unit_vectors_1x1(self):
        """Edge case: 1x1 image."""
        rays = compute_erp_ray_directions(h=1, w=1)
        assert rays.shape == (1, 1, 3)
        norm = np.linalg.norm(rays[0, 0])
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_top_row_y_near_one(self):
        """Top row (v=0, phi=0) should point mostly in the +y direction."""
        rays = compute_erp_ray_directions(h=64, w=128)
        top_y = rays[0, :, 1]  # y component of top row
        assert np.all(top_y > 0.99), "Top row should have y ~ 1 (north pole)"

    def test_bottom_row_y_near_minus_one(self):
        """Bottom row (v=h-1, phi=pi) should point mostly in the -y direction."""
        rays = compute_erp_ray_directions(h=64, w=128)
        bottom_y = rays[-1, :, 1]
        assert np.all(bottom_y < -0.99), "Bottom row should have y ~ -1 (south pole)"

    def test_equator_y_near_zero(self):
        """Middle row should have y ~ 0 (equator, phi=pi/2)."""
        # Use odd height so there is an exact middle row at phi = pi/2.
        h = 65
        rays = compute_erp_ray_directions(h=h, w=128)
        mid = h // 2
        equator_y = rays[mid, :, 1]
        np.testing.assert_allclose(equator_y, 0.0, atol=0.05)

    def test_varying_sizes(self):
        """Various sizes should all produce valid shapes and unit norms."""
        for h, w in [(8, 16), (100, 200), (3, 6)]:
            rays = compute_erp_ray_directions(h=h, w=w)
            assert rays.shape == (h, w, 3)
            norms = np.linalg.norm(rays, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5,
                                       err_msg=f"Failed for h={h}, w={w}")


class TestSeedGapGaussians:
    """Tests for seed_gap_gaussians."""

    def _make_depth_and_mask(self, h=64, w=128, depth_val=5.0, gap_fraction=0.5):
        depth = np.full((h, w), depth_val, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        # Mark the left half as gaps.
        mask[:, : w // 2] = True
        return depth, mask

    # --- Output structure ---

    def test_output_keys(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert "positions" in result
        assert "opacities" in result
        assert "provenance" in result

    def test_provenance_tag_value(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert result["provenance"] == "gap_seed"

    def test_positions_3d(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert result["positions"].ndim == 2
        assert result["positions"].shape[1] == 3

    def test_opacities_1d(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert result["opacities"].ndim == 1

    def test_positions_opacities_same_length(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert result["positions"].shape[0] == result["opacities"].shape[0]

    # --- Opacity value ---

    def test_default_opacity_value(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask, initial_opacity=0.01)
        np.testing.assert_allclose(result["opacities"], 0.01, atol=1e-7)

    def test_custom_opacity_value(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask, initial_opacity=0.5)
        np.testing.assert_allclose(result["opacities"], 0.5, atol=1e-7)

    # --- Empty gap case ---

    def test_no_gaps_returns_empty_positions(self):
        h, w = 32, 64
        depth = np.full((h, w), 3.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)  # all False — no gaps
        result = seed_gap_gaussians(depth, mask)
        assert result["positions"].shape == (0, 3)
        assert result["opacities"].shape == (0,)
        assert result["provenance"] == "gap_seed"

    # --- Stride ---

    def test_stride_reduces_count(self):
        h, w = 64, 128
        depth = np.full((h, w), 5.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)  # all gaps

        result_s1 = seed_gap_gaussians(depth, mask, stride=1)
        result_s4 = seed_gap_gaussians(depth, mask, stride=4)

        count_s1 = result_s1["positions"].shape[0]
        count_s4 = result_s4["positions"].shape[0]
        assert count_s4 < count_s1, (
            f"stride=4 ({count_s4}) should yield fewer Gaussians than stride=1 ({count_s1})"
        )

    def test_stride_1_uses_all_gap_pixels(self):
        h, w = 8, 16
        depth = np.full((h, w), 2.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)

        result = seed_gap_gaussians(depth, mask, stride=1)
        assert result["positions"].shape[0] == h * w

    def test_stride_2_approximately_quarters_count(self):
        h, w = 64, 128
        depth = np.full((h, w), 1.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)

        result_s1 = seed_gap_gaussians(depth, mask, stride=1)
        result_s2 = seed_gap_gaussians(depth, mask, stride=2)

        count_s1 = result_s1["positions"].shape[0]
        count_s2 = result_s2["positions"].shape[0]
        # stride=2 gives roughly 1/4 of pixels, ±rounding.
        ratio = count_s2 / count_s1
        assert 0.2 < ratio < 0.35, f"Expected ~0.25 ratio, got {ratio}"

    # --- Depth scaling ---

    def test_positions_scale_with_depth(self):
        """Positions at greater depth should be farther from the origin."""
        h, w = 32, 64
        mask = np.ones((h, w), dtype=bool)

        depth_near = np.full((h, w), 1.0, dtype=np.float32)
        depth_far = np.full((h, w), 10.0, dtype=np.float32)

        result_near = seed_gap_gaussians(depth_near, mask, stride=1)
        result_far = seed_gap_gaussians(depth_far, mask, stride=1)

        dist_near = np.linalg.norm(result_near["positions"], axis=-1).mean()
        dist_far = np.linalg.norm(result_far["positions"], axis=-1).mean()

        assert dist_far > dist_near, (
            f"Far depth ({dist_far:.3f}) should be farther from origin than near ({dist_near:.3f})"
        )

    def test_positions_magnitude_matches_depth(self):
        """Positions magnitude should equal depth for uniform depth maps."""
        h, w = 32, 64
        depth_val = 7.5
        depth = np.full((h, w), depth_val, dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)

        result = seed_gap_gaussians(depth, mask, stride=1)
        magnitudes = np.linalg.norm(result["positions"], axis=-1)
        # Rays are unit vectors, so positions = ray * depth => |position| == depth.
        np.testing.assert_allclose(magnitudes, depth_val, atol=1e-4)

    # --- dtype ---

    def test_positions_dtype_float32(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert result["positions"].dtype == np.float32

    def test_opacities_dtype_float32(self):
        depth, mask = self._make_depth_and_mask()
        result = seed_gap_gaussians(depth, mask)
        assert result["opacities"].dtype == np.float32
