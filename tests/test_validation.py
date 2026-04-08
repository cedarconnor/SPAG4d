"""Tests for spag4d.refine.validation."""

import numpy as np
import pytest

from spag4d.refine.validation import (
    AnchorCheckResult,
    ValidationReport,
    compute_coverage,
    compute_psnr,
    check_source_anchor,
)


# ---------------------------------------------------------------------------
# Tests for compute_psnr
# ---------------------------------------------------------------------------

class TestComputePsnr:
    def test_identical_images(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        psnr = compute_psnr(img, img)
        assert psnr > 50

    def test_identical_images_capped(self):
        """Identical images should hit the 100 dB cap."""
        img = np.zeros((64, 64, 3), dtype=np.float32)
        assert compute_psnr(img, img) == pytest.approx(100.0)

    def test_different_images(self):
        a = np.zeros((64, 64, 3), dtype=np.float32)
        b = np.ones((64, 64, 3), dtype=np.float32)
        psnr = compute_psnr(a, b)
        assert psnr == pytest.approx(0.0, abs=0.1)

    def test_similar_images(self):
        rng = np.random.default_rng(42)
        a = rng.random((64, 64, 3)).astype(np.float32)
        b = np.clip(a + rng.random((64, 64, 3)).astype(np.float32) * 0.01, 0, 1)
        psnr = compute_psnr(a, b)
        assert 30 < psnr < 60

    def test_returns_float(self):
        a = np.zeros((8, 8), dtype=np.float32)
        b = np.full((8, 8), 0.1, dtype=np.float32)
        result = compute_psnr(a, b)
        assert isinstance(result, float)

    def test_grayscale_input(self):
        """Should work on 2-D arrays as well as 3-D."""
        a = np.full((32, 32), 0.5, dtype=np.float32)
        b = np.full((32, 32), 0.6, dtype=np.float32)
        psnr = compute_psnr(a, b)
        assert psnr > 0


# ---------------------------------------------------------------------------
# Tests for compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverage:
    def test_full_coverage(self):
        masks = [np.zeros((64, 64), dtype=np.float32) for _ in range(5)]
        assert compute_coverage(masks) == pytest.approx(1.0)

    def test_no_coverage(self):
        masks = [np.ones((64, 64), dtype=np.float32) for _ in range(5)]
        assert compute_coverage(masks) == pytest.approx(0.0)

    def test_half_coverage(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[:32, :] = 1.0
        masks = [mask.copy() for _ in range(5)]
        assert compute_coverage(masks) == pytest.approx(0.5, abs=0.01)

    def test_empty_list(self):
        assert compute_coverage([]) == pytest.approx(1.0)

    def test_single_mask(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[:4, :] = 1.0  # 25 % holes
        assert compute_coverage([mask]) == pytest.approx(0.75, abs=0.01)

    def test_returns_float(self):
        masks = [np.zeros((8, 8), dtype=np.float32)]
        result = compute_coverage(masks)
        assert isinstance(result, float)

    def test_mixed_masks(self):
        full = np.zeros((64, 64), dtype=np.float32)   # coverage = 1.0
        empty = np.ones((64, 64), dtype=np.float32)   # coverage = 0.0
        assert compute_coverage([full, empty]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests for check_source_anchor
# ---------------------------------------------------------------------------

class TestCheckSourceAnchor:
    def test_pass(self):
        result = check_source_anchor(baseline_psnr=30.0, current_psnr=31.0, floor=25.0)
        assert result.passed is True
        assert result.degradation == pytest.approx(0.0)

    def test_pass_no_degradation(self):
        """Equal PSNR should pass with degradation = 0."""
        result = check_source_anchor(baseline_psnr=28.0, current_psnr=28.0, floor=25.0)
        assert result.passed is True
        assert result.degradation == pytest.approx(0.0)

    def test_fail_below_floor(self):
        result = check_source_anchor(baseline_psnr=30.0, current_psnr=24.0, floor=25.0)
        assert result.passed is False

    def test_fail_degradation(self):
        result = check_source_anchor(baseline_psnr=35.0, current_psnr=30.0, floor=25.0)
        assert result.passed is False
        assert result.degradation == pytest.approx(5.0)

    def test_fail_both_conditions(self):
        """Below floor AND high degradation should still fail."""
        result = check_source_anchor(baseline_psnr=40.0, current_psnr=20.0, floor=25.0)
        assert result.passed is False
        assert result.degradation == pytest.approx(20.0)

    def test_degradation_clamped_at_zero_for_improvement(self):
        """When current > baseline, degradation is 0 not negative."""
        result = check_source_anchor(baseline_psnr=25.0, current_psnr=30.0, floor=20.0)
        assert result.degradation == pytest.approx(0.0)
        assert result.passed is True

    def test_result_fields(self):
        result = check_source_anchor(baseline_psnr=30.0, current_psnr=29.0, floor=25.0)
        assert result.baseline_psnr == pytest.approx(30.0)
        assert result.current_psnr == pytest.approx(29.0)
        assert result.floor == pytest.approx(25.0)

    def test_custom_max_degradation_passes(self):
        result = check_source_anchor(
            baseline_psnr=35.0, current_psnr=33.0, floor=25.0, max_degradation=3.0
        )
        assert result.passed is True

    def test_custom_max_degradation_fails(self):
        result = check_source_anchor(
            baseline_psnr=35.0, current_psnr=33.0, floor=25.0, max_degradation=1.0
        )
        assert result.passed is False

    def test_returns_anchor_check_result(self):
        result = check_source_anchor(baseline_psnr=30.0, current_psnr=30.0)
        assert isinstance(result, AnchorCheckResult)


# ---------------------------------------------------------------------------
# Smoke tests for dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_anchor_check_result_fields(self):
        r = AnchorCheckResult(
            passed=True, baseline_psnr=30.0, current_psnr=31.0, floor=25.0, degradation=0.0
        )
        assert r.passed is True

    def test_validation_report_fields(self):
        anchor = AnchorCheckResult(
            passed=True, baseline_psnr=30.0, current_psnr=31.0, floor=25.0, degradation=0.0
        )
        report = ValidationReport(
            anchor_check=anchor,
            coverage_before=0.7,
            coverage_after=0.9,
            coverage_improvement=0.2,
            iteration=3,
        )
        assert report.iteration == 3
        assert report.coverage_improvement == pytest.approx(0.2)
