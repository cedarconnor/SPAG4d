# tests/test_scene_filter.py
"""
Tests for Phase 3: Sky detection and pole-density normalisation.

Covers scene_filter.py functions:
  - detect_sky_depth / detect_sky_gradient
  - compute_pole_thinning_mask
  - filter_gaussian_candidates
  - apply_sky_mode_to_gaussians
"""

import numpy as np
import pytest
import torch


# ── Sky Detection ─────────────────────────────────────────────────────────────

class TestDetectSkyDepth:
    def test_output_shape(self):
        from spag4d.scene_filter import detect_sky_depth
        depth = np.random.uniform(1.0, 120.0, (64, 128)).astype(np.float32)
        mask = detect_sky_depth(depth, depth_max=100.0)
        assert mask.shape == (64, 128)
        assert mask.dtype == bool

    def test_above_threshold_is_sky(self):
        """Pixels > depth_max * 0.90 are classified as sky."""
        from spag4d.scene_filter import detect_sky_depth
        depth = np.full((32, 64), 95.0, dtype=np.float32)  # 95% of 100.0 → sky
        mask = detect_sky_depth(depth, depth_max=100.0, threshold_ratio=0.90)
        assert mask.all(), "All near-max pixels should be sky"

    def test_below_threshold_is_not_sky(self):
        from spag4d.scene_filter import detect_sky_depth
        depth = np.full((32, 64), 5.0, dtype=np.float32)
        mask = detect_sky_depth(depth, depth_max=100.0, threshold_ratio=0.90)
        assert not mask.any(), "Near-camera pixels should not be sky"

    def test_mixed_scene(self):
        """Half the image is sky, half is foreground."""
        from spag4d.scene_filter import detect_sky_depth
        H, W = 64, 128
        depth = np.full((H, W), 5.0, dtype=np.float32)
        depth[:, W // 2:] = 95.0   # right half is sky
        mask = detect_sky_depth(depth, depth_max=100.0, threshold_ratio=0.90)
        assert not mask[:, :W // 2].any(), "Left foreground should not be sky"
        assert mask[:, W // 2:].all(),     "Right sky region should be sky"


class TestDetectSkyGradient:
    def test_output_shape_and_dtype(self):
        from spag4d.scene_filter import detect_sky_gradient
        H, W = 64, 128
        depth = np.random.uniform(1.0, 60.0, (H, W)).astype(np.float32)
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        mask = detect_sky_gradient(depth, image, depth_max=100.0)
        assert mask.shape == (H, W)
        assert mask.dtype == bool

    def test_no_false_sky_in_uniform_near_depth(self):
        """Low, uniform depth with no gradient → no sky pixels."""
        from spag4d.scene_filter import detect_sky_gradient
        H, W = 32, 64
        depth = np.full((H, W), 5.0, dtype=np.float32)
        image = np.full((H, W, 3), 128, dtype=np.uint8)
        mask = detect_sky_gradient(depth, image, depth_max=100.0)
        # Must not classify any near-camera pixel as sky
        assert not mask.any(), "Near, uniform surface should have no sky detections"

    def test_sky_region_detected_more_than_foreground(self):
        """detect_sky_gradient detects sky more in the sky half than the foreground half.

        The function uses percentile thresholds over the entire image, so it's designed
        for mixed scenes. We verify that the sky half (high depth, uniform) gets
        classified as sky more often than the foreground half (low depth, textured).
        """
        from spag4d.scene_filter import detect_sky_gradient
        H, W = 128, 256
        rng = np.random.default_rng(42)

        # Top half: sky — high depth, near-zero gradient, uniform blue-gray color
        sky_depth = np.full((H // 2, W), 80.0, dtype=np.float32)
        sky_depth += rng.uniform(0, 0.5, sky_depth.shape).astype(np.float32)
        sky_color = np.clip(
            np.full((H // 2, W, 3), 200) + rng.integers(-2, 3, (H // 2, W, 3)),
            0, 255
        ).astype(np.uint8)

        # Bottom half: foreground — varied depth, high gradient, textured color
        fg_depth = rng.uniform(1.0, 20.0, (H // 2, W)).astype(np.float32)
        fg_color = rng.integers(0, 255, (H // 2, W, 3), dtype=np.uint8)

        depth = np.vstack([sky_depth, fg_depth])
        image = np.vstack([sky_color, fg_color])

        mask = detect_sky_gradient(depth, image, depth_max=100.0)

        sky_rate = float(mask[:H // 2].mean())
        fg_rate  = float(mask[H // 2:].mean())

        assert sky_rate > fg_rate, (
            f"Sky region rate ({sky_rate:.2%}) should exceed foreground rate ({fg_rate:.2%})"
        )


# ── Pole Thinning ─────────────────────────────────────────────────────────────

class TestComputePoleThinningMask:
    def test_output_shape(self):
        from spag4d.scene_filter import compute_pole_thinning_mask
        mask = compute_pole_thinning_mask(H=64, W=128, stride=2)
        expected_rows = len(range(0, 64, 2))
        expected_cols = len(range(0, 128, 2))
        assert mask.shape == (expected_rows, expected_cols)
        assert mask.dtype == bool

    def test_equator_higher_keep_prob_than_poles(self):
        """sin(θ) → equator rows have higher keep rate than pole rows."""
        from spag4d.scene_filter import compute_pole_thinning_mask
        # Use a large grid so averages are stable
        H, W, stride = 512, 1024, 1
        mask = compute_pole_thinning_mask(H=H, W=W, stride=stride, seed=0)
        n_rows = mask.shape[0]
        equator_band = slice(n_rows * 4 // 10, n_rows * 6 // 10)  # 40–60%
        pole_band    = slice(0, n_rows // 10)                      # top 10%

        equator_keep = float(mask[equator_band].mean())
        pole_keep    = float(mask[pole_band].mean())
        assert equator_keep > pole_keep, (
            f"Equator keep={equator_keep:.3f} should exceed pole keep={pole_keep:.3f}"
        )

    def test_min_density_ratio_enforced(self):
        """No row should have zero keep probability (min_density_ratio floor)."""
        from spag4d.scene_filter import compute_pole_thinning_mask
        # With many samples and a high min_density_ratio, pole rows must have keeps
        mask = compute_pole_thinning_mask(H=512, W=1024, stride=1,
                                          min_density_ratio=0.50, seed=1)
        # Even the first (pole) row must have >0 kept pixels
        assert mask[0].any(), "Pole row should still have some kept pixels"

    def test_reproducible_with_same_seed(self):
        from spag4d.scene_filter import compute_pole_thinning_mask
        m1 = compute_pole_thinning_mask(H=64, W=128, stride=2, seed=42)
        m2 = compute_pole_thinning_mask(H=64, W=128, stride=2, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_differ(self):
        from spag4d.scene_filter import compute_pole_thinning_mask
        m1 = compute_pole_thinning_mask(H=128, W=256, stride=2, seed=1)
        m2 = compute_pole_thinning_mask(H=128, W=256, stride=2, seed=2)
        assert not np.array_equal(m1, m2)


# ── filter_gaussian_candidates ────────────────────────────────────────────────

class TestFilterGaussianCandidates:
    def _make_inputs(self, H=64, W=128, sky_depth=95.0, fg_depth=5.0):
        depth = np.full((H, W), fg_depth, dtype=np.float32)
        depth[:, W // 2:] = sky_depth
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        return depth, image

    def test_output_types_and_shapes(self):
        from spag4d.scene_filter import filter_gaussian_candidates, SkyMode
        depth, image = self._make_inputs()
        stride = 2
        keep, sky = filter_gaussian_candidates(depth, image, stride=stride,
                                               sky_mode=SkyMode.SKIP,
                                               sky_detection="depth")
        H_s = len(range(0, 64, stride))
        W_s = len(range(0, 128, stride))
        assert keep.shape == (H_s, W_s)
        assert sky.shape  == (H_s, W_s)
        assert keep.dtype == bool
        assert sky.dtype  == bool

    def test_depth_sky_excluded_from_keep(self):
        """Pixels classified as sky should not appear in keep_mask (SKIP mode)."""
        from spag4d.scene_filter import filter_gaussian_candidates, SkyMode
        depth, image = self._make_inputs(sky_depth=96.0, fg_depth=5.0)
        keep, sky = filter_gaussian_candidates(depth, image, stride=1,
                                               sky_mode=SkyMode.SKIP,
                                               sky_detection="depth",
                                               depth_max=100.0)
        # Sky pixels must not appear in keep_mask
        overlap = keep & sky
        assert not overlap.any(), "Sky pixels should be excluded from keep_mask"

    def test_depth_range_filtering(self):
        """Pixels outside [depth_min, depth_max] must be excluded from keep."""
        from spag4d.scene_filter import filter_gaussian_candidates, SkyMode
        H, W = 32, 64
        depth = np.full((H, W), 5.0, dtype=np.float32)
        depth[:H // 2, :] = 0.01   # too close → invalid
        image = np.zeros((H, W, 3), dtype=np.uint8)
        keep, _ = filter_gaussian_candidates(depth, image, stride=1,
                                             sky_mode=SkyMode.SKIP,
                                             sky_detection="none",
                                             pole_thinning=False,
                                             depth_min=0.1, depth_max=100.0)
        # Top half (depth < 0.1) must be excluded
        assert not keep[:H // 2, :].any(), "Sub-depth_min pixels must be excluded"
        assert keep[H // 2:, :].any(),      "Valid-depth pixels should be included"

    def test_no_sky_gaussians_at_sky_depth(self):
        """keep_mask positions should have depth < depth_max * threshold in SKIP mode."""
        from spag4d.scene_filter import filter_gaussian_candidates, SkyMode
        H, W = 64, 128
        depth_max = 100.0
        depth = np.random.uniform(1.0, 50.0, (H, W)).astype(np.float32)
        depth[:H // 4, :] = 96.0   # top quarter is sky
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        keep, _ = filter_gaussian_candidates(
            depth, image, stride=1,
            sky_mode=SkyMode.SKIP,
            sky_detection="depth",
            pole_thinning=False,
            depth_min=0.1, depth_max=depth_max,
        )
        # Depth at all kept positions should be below the sky threshold
        kept_depths = depth[keep]
        sky_thresh = depth_max * 0.90
        assert (kept_depths < sky_thresh).all(), (
            "keep_mask must not include any pixel at sky depth"
        )


# ── apply_sky_mode_to_gaussians ───────────────────────────────────────────────

class TestApplySkyModeToGaussians:
    def _make_gaussians(self, n=100, device=None):
        if device is None:
            device = torch.device("cpu")
        return {
            "means":     torch.randn(n, 3),
            "scales":    torch.rand(n, 3) * 0.1 + 0.01,
            "quats":     torch.randn(n, 4),
            "colors":    torch.rand(n, 3),
            "opacities": torch.rand(n, 1) * 0.5 + 0.5,
            "sh1":       torch.zeros(n, 9),
        }

    def test_skip_mode_returns_unchanged(self):
        from spag4d.scene_filter import apply_sky_mode_to_gaussians, SkyMode
        base = self._make_gaussians(100)
        sky_mask = np.ones((16, 32), dtype=bool)
        depth_s  = np.full((16, 32), 5.0, dtype=np.float32)
        image_s  = np.zeros((16, 32, 3), dtype=np.uint8)
        result = apply_sky_mode_to_gaussians(
            base, sky_mask, depth_s, image_s, SkyMode.SKIP
        )
        assert result is base, "SKIP mode must return the original dict unchanged"

    def test_background_sphere_appends_gaussians(self):
        from spag4d.scene_filter import apply_sky_mode_to_gaussians, SkyMode
        base = self._make_gaussians(100)
        # 20% of the grid is sky
        sky_mask = np.zeros((16, 32), dtype=bool)
        sky_mask[:4, :] = True
        depth_s = np.full((16, 32), 90.0, dtype=np.float32)
        image_s = np.full((16, 32, 3), 120, dtype=np.uint8)

        result = apply_sky_mode_to_gaussians(
            base, sky_mask, depth_s, image_s, SkyMode.BACKGROUND_SPHERE,
            sky_radius=200.0
        )
        n_sky = int(sky_mask.sum())
        assert result["means"].shape[0] == 100 + n_sky, (
            "BACKGROUND_SPHERE should append one Gaussian per sky pixel"
        )

    def test_low_opacity_mode_appends_gaussians(self):
        from spag4d.scene_filter import apply_sky_mode_to_gaussians, SkyMode
        base = self._make_gaussians(50)
        sky_mask = np.zeros((8, 16), dtype=bool)
        sky_mask[0, :] = True  # 16 sky pixels
        depth_s = np.full((8, 16), 90.0, dtype=np.float32)
        image_s = np.zeros((8, 16, 3), dtype=np.uint8)

        result = apply_sky_mode_to_gaussians(
            base, sky_mask, depth_s, image_s, SkyMode.LOW_OPACITY, sky_opacity=0.15
        )
        n_sky = int(sky_mask.sum())
        assert result["means"].shape[0] == 50 + n_sky

        # Sky Gaussians should have low opacity
        sky_opacities = result["opacities"][50:]
        assert (sky_opacities <= 0.16).all(), "Sky Gaussians should have low opacity"

    def test_no_sky_pixels_returns_base_unchanged(self):
        from spag4d.scene_filter import apply_sky_mode_to_gaussians, SkyMode
        base = self._make_gaussians(80)
        sky_mask = np.zeros((8, 16), dtype=bool)   # no sky
        depth_s  = np.full((8, 16), 5.0, dtype=np.float32)
        image_s  = np.zeros((8, 16, 3), dtype=np.uint8)

        result = apply_sky_mode_to_gaussians(
            base, sky_mask, depth_s, image_s, SkyMode.BACKGROUND_SPHERE
        )
        assert result["means"].shape[0] == 80, "No sky pixels → no Gaussians appended"

    def test_sphere_gaussians_are_at_sky_radius(self):
        """BACKGROUND_SPHERE mode places Gaussians at the sky_radius distance."""
        from spag4d.scene_filter import apply_sky_mode_to_gaussians, SkyMode
        base = self._make_gaussians(10)
        sky_mask = np.zeros((4, 8), dtype=bool)
        sky_mask[:, :4] = True  # 16 sky pixels
        depth_s  = np.full((4, 8), 90.0, dtype=np.float32)
        image_s  = np.zeros((4, 8, 3), dtype=np.uint8)
        sky_radius = 200.0

        result = apply_sky_mode_to_gaussians(
            base, sky_mask, depth_s, image_s,
            SkyMode.BACKGROUND_SPHERE, sky_radius=sky_radius
        )
        # Sky Gaussians are appended at the end
        sky_means = result["means"][10:]
        radii = sky_means.norm(dim=-1)
        assert torch.allclose(radii, torch.full_like(radii, sky_radius), atol=1e-3)


# ── Integration: pole density uniformity ─────────────────────────────────────

class TestPoleDensityUniformity:
    def test_cv_of_density_per_latitude_band(self):
        """
        Coefficient of variation (CV) of Gaussian density per latitude band
        should be < 0.5 after pole thinning (much lower than without thinning).
        """
        from spag4d.scene_filter import compute_pole_thinning_mask

        H, W, stride = 512, 1024, 2
        mask = compute_pole_thinning_mask(H=H, W=W, stride=stride, seed=0)
        n_rows = mask.shape[0]

        # Divide into 10 latitude bands and count kept pixels per band
        band_size = n_rows // 10
        counts = []
        for b in range(10):
            band = mask[b * band_size : (b + 1) * band_size]
            counts.append(float(band.sum()))

        counts = np.array(counts)
        cv = counts.std() / (counts.mean() + 1e-8)

        # CV < 0.5 means reasonable uniformity (perfect ERP has CV → ∞ without thinning)
        assert cv < 0.5, f"Latitude density CV={cv:.3f} is too high (pole blob present)"
