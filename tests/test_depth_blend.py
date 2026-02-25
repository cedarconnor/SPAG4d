# tests/test_depth_blend.py
"""
Tests for Phase 4: Depth map fusion and blending utilities.

Covers depth_blend.py:
  - _build_gaussian_pyramid / _build_laplacian_pyramid / _reconstruct_from_laplacian
  - laplacian_depth_fusion
  - masked_laplacian_fusion
  - feathered_blend
  - DepthBlender (FEATHERED and LAPLACIAN modes)
"""

import numpy as np
import pytest


# ── pyramid helpers ───────────────────────────────────────────────────────────

class TestBuildGaussianPyramid:
    def test_level_count(self):
        from spag4d.depth_blend import _build_gaussian_pyramid
        img = np.random.uniform(1.0, 10.0, (64, 128)).astype(np.float32)
        pyr = _build_gaussian_pyramid(img, n_levels=4)
        assert len(pyr) == 4

    def test_first_level_matches_input_dtype(self):
        from spag4d.depth_blend import _build_gaussian_pyramid
        img = np.ones((32, 64), dtype=np.float64) * 5.0
        pyr = _build_gaussian_pyramid(img, n_levels=3)
        assert pyr[0].dtype == np.float32

    def test_progressive_downsampling(self):
        """Each level should be ~half the resolution of the previous."""
        from spag4d.depth_blend import _build_gaussian_pyramid
        H, W = 128, 256
        img = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        pyr = _build_gaussian_pyramid(img, n_levels=4)
        for i in range(1, len(pyr)):
            ratio_h = pyr[i - 1].shape[0] / pyr[i].shape[0]
            ratio_w = pyr[i - 1].shape[1] / pyr[i].shape[1]
            assert 1.8 <= ratio_h <= 2.2, f"Level {i} H ratio {ratio_h:.2f} unexpected"
            assert 1.8 <= ratio_w <= 2.2, f"Level {i} W ratio {ratio_w:.2f} unexpected"

    def test_single_level_equals_input(self):
        from spag4d.depth_blend import _build_gaussian_pyramid
        img = np.random.uniform(0.0, 5.0, (16, 32)).astype(np.float32)
        pyr = _build_gaussian_pyramid(img, n_levels=1)
        assert len(pyr) == 1
        np.testing.assert_array_equal(pyr[0], img)


class TestBuildLaplacianPyramid:
    def test_level_count_matches_gaussian(self):
        from spag4d.depth_blend import _build_gaussian_pyramid, _build_laplacian_pyramid
        img = np.random.uniform(1.0, 10.0, (64, 128)).astype(np.float32)
        gpyr = _build_gaussian_pyramid(img, n_levels=5)
        lpyr = _build_laplacian_pyramid(gpyr)
        assert len(lpyr) == len(gpyr)

    def test_first_level_shape_matches_input(self):
        from spag4d.depth_blend import _build_gaussian_pyramid, _build_laplacian_pyramid
        H, W = 64, 128
        img = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        gpyr = _build_gaussian_pyramid(img, n_levels=4)
        lpyr = _build_laplacian_pyramid(gpyr)
        assert lpyr[0].shape == (H, W)

    def test_coarsest_level_equals_gaussian_coarsest(self):
        """Coarsest Laplacian level must be the Gaussian residual (no subtraction)."""
        from spag4d.depth_blend import _build_gaussian_pyramid, _build_laplacian_pyramid
        img = np.random.uniform(1.0, 5.0, (32, 64)).astype(np.float32)
        gpyr = _build_gaussian_pyramid(img, n_levels=3)
        lpyr = _build_laplacian_pyramid(gpyr)
        np.testing.assert_array_equal(lpyr[-1], gpyr[-1])


class TestReconstructFromLaplacian:
    def test_round_trip_smooth_image(self):
        """Gaussian → Laplacian → reconstruct ≈ original (within float precision)."""
        from spag4d.depth_blend import (
            _build_gaussian_pyramid, _build_laplacian_pyramid,
            _reconstruct_from_laplacian,
        )
        img = np.random.uniform(1.0, 10.0, (64, 128)).astype(np.float32)
        gpyr = _build_gaussian_pyramid(img, n_levels=5)
        lpyr = _build_laplacian_pyramid(gpyr)
        recovered = _reconstruct_from_laplacian(lpyr)
        # Allow some float32 rounding accumulated across levels
        np.testing.assert_allclose(recovered, img, atol=1e-2, rtol=0)

    def test_output_shape_matches_finest_level(self):
        from spag4d.depth_blend import (
            _build_gaussian_pyramid, _build_laplacian_pyramid,
            _reconstruct_from_laplacian,
        )
        H, W = 64, 128
        img = np.random.uniform(0.0, 5.0, (H, W)).astype(np.float32)
        gpyr = _build_gaussian_pyramid(img, n_levels=4)
        lpyr = _build_laplacian_pyramid(gpyr)
        out = _reconstruct_from_laplacian(lpyr)
        assert out.shape == (H, W)


# ── laplacian_depth_fusion ─────────────────────────────────────────────────────

class TestLaplacianDepthFusion:
    def test_output_shape_and_dtype(self):
        from spag4d.depth_blend import laplacian_depth_fusion
        H, W = 64, 128
        dap = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        dp  = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        fused = laplacian_depth_fusion(dap, dp)
        assert fused.shape == (H, W)
        assert fused.dtype == np.float32

    def test_output_non_negative(self):
        """Fused depth must never go negative."""
        from spag4d.depth_blend import laplacian_depth_fusion
        dap = np.random.uniform(0.1, 20.0, (64, 128)).astype(np.float32)
        dp  = np.random.uniform(0.1, 20.0, (64, 128)).astype(np.float32)
        fused = laplacian_depth_fusion(dap, dp)
        assert fused.min() >= 0.0

    def test_identical_inputs_returns_same(self):
        """When both inputs are the same, fusion should return the same depth."""
        from spag4d.depth_blend import laplacian_depth_fusion
        depth = np.ones((32, 64), dtype=np.float32) * 5.0
        fused = laplacian_depth_fusion(depth, depth, n_levels=4)
        np.testing.assert_allclose(fused, depth, atol=0.05)

    def test_low_freq_dominated_by_dap(self):
        """With low_freq_cutoff=n_levels-1, all coarse bands come from dap_depth."""
        from spag4d.depth_blend import laplacian_depth_fusion
        H, W = 64, 128
        dap = np.linspace(1, 10, W).reshape(1, -1).repeat(H, axis=0).astype(np.float32)
        dp  = np.random.uniform(5.0, 6.0, (H, W)).astype(np.float32)
        # cutoff at 3 out of 4 levels → fused should preserve the horizontal gradient of dap
        fused = laplacian_depth_fusion(dap, dp, n_levels=4, low_freq_cutoff=3)
        # Left edge should be close to 1, right edge close to 10
        assert fused[:, 0].mean() < 3.0, "Coarse structure lost on left edge"
        assert fused[:, -1].mean() > 8.0, "Coarse structure lost on right edge"

    def test_high_freq_dominated_by_dp(self):
        """With low_freq_cutoff=0, fine detail should come from dp_depth."""
        from spag4d.depth_blend import laplacian_depth_fusion
        H, W = 64, 128
        dap = np.full((H, W), 5.0, dtype=np.float32)
        dp  = np.zeros((H, W), dtype=np.float32)
        # Inject sharp high-frequency vertical stripe in dp
        dp[:, W//2] = 10.0
        # cutoff=0 → finest detail levels come from dp, base structure from dap
        fused = laplacian_depth_fusion(dap, dp, n_levels=4, low_freq_cutoff=0)
        # the center stripe from dp should heavily influence the output
        assert fused[:, W//2].mean() > 6.0, "High frequency detail from dp was lost"
        assert fused[:, 0].mean() < 6.0, "Low frequency structure incorrectly came from dp"


# ── masked_laplacian_fusion ────────────────────────────────────────────────────

class TestMaskedLaplacianFusion:
    def test_output_shape_and_dtype(self):
        from spag4d.depth_blend import masked_laplacian_fusion
        H, W = 64, 128
        dap  = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        dp   = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        conf = np.random.uniform(0.0, 1.0,  (H, W)).astype(np.float32)
        fused = masked_laplacian_fusion(dap, dp, conf)
        assert fused.shape == (H, W)
        assert fused.dtype == np.float32

    def test_output_non_negative(self):
        from spag4d.depth_blend import masked_laplacian_fusion
        dap  = np.random.uniform(0.1, 20.0, (64, 128)).astype(np.float32)
        dp   = np.random.uniform(0.1, 20.0, (64, 128)).astype(np.float32)
        conf = np.random.uniform(0.0, 1.0,  (64, 128)).astype(np.float32)
        fused = masked_laplacian_fusion(dap, dp, conf)
        assert fused.min() >= 0.0

    def test_full_confidence_biases_toward_dp(self):
        """With confidence=1 everywhere, fused depth should be close to dp_depth."""
        from spag4d.depth_blend import masked_laplacian_fusion
        H, W = 64, 128
        dap  = np.full((H, W), 2.0, dtype=np.float32)
        dp   = np.full((H, W), 9.0, dtype=np.float32)
        conf = np.ones((H, W), dtype=np.float32)
        fused = masked_laplacian_fusion(dap, dp, conf)
        assert fused.mean() > 6.0, (
            f"Full-confidence fused mean {fused.mean():.2f} should be close to dp (9.0)"
        )

    def test_zero_confidence_biases_toward_dap(self):
        """With confidence=0 everywhere, fused depth should be close to dap_depth."""
        from spag4d.depth_blend import masked_laplacian_fusion
        H, W = 64, 128
        dap  = np.full((H, W), 2.0, dtype=np.float32)
        dp   = np.full((H, W), 9.0, dtype=np.float32)
        conf = np.zeros((H, W), dtype=np.float32)
        fused = masked_laplacian_fusion(dap, dp, conf)
        assert fused.mean() < 4.0, (
            f"Zero-confidence fused mean {fused.mean():.2f} should be close to dap (2.0)"
        )


# ── feathered_blend ────────────────────────────────────────────────────────────

class TestFeatheredBlend:
    def test_weight_one_returns_depth_a(self):
        from spag4d.depth_blend import feathered_blend
        H, W = 32, 64
        a = np.full((H, W), 5.0, dtype=np.float32)
        b = np.full((H, W), 10.0, dtype=np.float32)
        w = np.ones((H, W), dtype=np.float32)
        result = feathered_blend(a, b, w)
        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_weight_zero_returns_depth_b(self):
        from spag4d.depth_blend import feathered_blend
        H, W = 32, 64
        a = np.full((H, W), 5.0, dtype=np.float32)
        b = np.full((H, W), 10.0, dtype=np.float32)
        w = np.zeros((H, W), dtype=np.float32)
        result = feathered_blend(a, b, w)
        np.testing.assert_allclose(result, b, atol=1e-6)

    def test_half_weight_is_average(self):
        from spag4d.depth_blend import feathered_blend
        H, W = 16, 32
        a = np.full((H, W), 4.0, dtype=np.float32)
        b = np.full((H, W), 8.0, dtype=np.float32)
        w = np.full((H, W), 0.5, dtype=np.float32)
        result = feathered_blend(a, b, w)
        np.testing.assert_allclose(result, 6.0, atol=1e-6)

    def test_output_shape_and_dtype(self):
        from spag4d.depth_blend import feathered_blend
        H, W = 64, 128
        a = np.random.uniform(1.0, 5.0, (H, W)).astype(np.float32)
        b = np.random.uniform(5.0, 10.0, (H, W)).astype(np.float32)
        w = np.random.uniform(0.0, 1.0, (H, W)).astype(np.float32)
        result = feathered_blend(a, b, w)
        assert result.shape == (H, W)
        assert result.dtype == np.float32

    def test_weight_clipped_outside_unit_interval(self):
        """Weights outside [0,1] should be clipped, not extrapolate."""
        from spag4d.depth_blend import feathered_blend
        a = np.full((4, 4), 2.0, dtype=np.float32)
        b = np.full((4, 4), 8.0, dtype=np.float32)
        w_high = np.full((4, 4), 2.0, dtype=np.float32)   # > 1 → clipped to 1
        w_low  = np.full((4, 4), -1.0, dtype=np.float32)  # < 0 → clipped to 0
        np.testing.assert_allclose(feathered_blend(a, b, w_high), a, atol=1e-6)
        np.testing.assert_allclose(feathered_blend(a, b, w_low),  b, atol=1e-6)


# ── DepthBlender class ─────────────────────────────────────────────────────────

class TestDepthBlender:
    def test_feathered_mode_no_confidence(self):
        """FEATHERED mode without confidence → 50/50 blend."""
        from spag4d.depth_blend import DepthBlender, BlendMode
        blender = DepthBlender(mode=BlendMode.FEATHERED)
        dap = np.full((32, 64), 2.0, dtype=np.float32)
        dp  = np.full((32, 64), 8.0, dtype=np.float32)
        out = blender.fuse(dap, dp)
        np.testing.assert_allclose(out.mean(), 5.0, atol=0.1)

    def test_feathered_mode_with_confidence(self):
        """FEATHERED mode with confidence=1 → dp_depth dominates."""
        from spag4d.depth_blend import DepthBlender, BlendMode
        blender = DepthBlender(mode=BlendMode.FEATHERED)
        H, W = 32, 64
        dap  = np.full((H, W), 2.0, dtype=np.float32)
        dp   = np.full((H, W), 8.0, dtype=np.float32)
        conf = np.ones((H, W), dtype=np.float32)
        out = blender.fuse(dap, dp, confidence=conf)
        np.testing.assert_allclose(out.mean(), 8.0, atol=0.1)

    def test_laplacian_mode_output_shape(self):
        from spag4d.depth_blend import DepthBlender, BlendMode
        blender = DepthBlender(mode=BlendMode.LAPLACIAN)
        H, W = 64, 128
        dap = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        dp  = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        out = blender.fuse(dap, dp)
        assert out.shape == (H, W)

    def test_laplacian_mode_with_confidence_output_shape(self):
        from spag4d.depth_blend import DepthBlender, BlendMode
        blender = DepthBlender(mode=BlendMode.LAPLACIAN)
        H, W = 64, 128
        dap  = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        dp   = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        conf = np.random.uniform(0.0, 1.0,  (H, W)).astype(np.float32)
        out = blender.fuse(dap, dp, confidence=conf)
        assert out.shape == (H, W)

    def test_laplacian_output_non_negative(self):
        from spag4d.depth_blend import DepthBlender, BlendMode
        blender = DepthBlender(mode=BlendMode.LAPLACIAN)
        dap = np.random.uniform(0.1, 20.0, (64, 128)).astype(np.float32)
        dp  = np.random.uniform(0.1, 20.0, (64, 128)).astype(np.float32)
        out = blender.fuse(dap, dp)
        assert out.min() >= 0.0

    def test_poisson_mode_raises_for_fuse(self):
        """DepthBlender.fuse() with POISSON should raise ValueError."""
        from spag4d.depth_blend import DepthBlender, BlendMode
        blender = DepthBlender(mode=BlendMode.POISSON)
        dap = np.ones((16, 32), dtype=np.float32)
        dp  = np.ones((16, 32), dtype=np.float32) * 2.0
        with pytest.raises(ValueError, match="poisson_blend_faces"):
            blender.fuse(dap, dp)

    def test_n_levels_parameter_respected(self):
        """Changing n_levels should still produce valid output."""
        from spag4d.depth_blend import DepthBlender, BlendMode
        for n in (3, 5, 7):
            blender = DepthBlender(mode=BlendMode.LAPLACIAN, n_levels=n)
            dap = np.random.uniform(1.0, 5.0, (64, 128)).astype(np.float32)
            dp  = np.random.uniform(1.0, 5.0, (64, 128)).astype(np.float32)
            out = blender.fuse(dap, dp)
            assert out.shape == (64, 128), f"n_levels={n} gave wrong shape"
            assert out.min() >= 0.0


# ── blend_mode enum ────────────────────────────────────────────────────────────

class TestBlendModeEnum:
    def test_enum_values(self):
        from spag4d.depth_blend import BlendMode
        assert BlendMode.FEATHERED.value == "feathered"
        assert BlendMode.LAPLACIAN.value == "laplacian"
        assert BlendMode.POISSON.value == "poisson"

    def test_enum_from_string(self):
        from spag4d.depth_blend import BlendMode
        assert BlendMode("feathered") == BlendMode.FEATHERED
        assert BlendMode("laplacian") == BlendMode.LAPLACIAN
        assert BlendMode("poisson")   == BlendMode.POISSON
