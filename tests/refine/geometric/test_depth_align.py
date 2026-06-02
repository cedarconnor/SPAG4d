# tests/refine/geometric/test_depth_align.py
import numpy as np
import pytest
from spag4d.refine.geometric.depth_align import align_depth_irls, AlignmentResult


def _make_aligned_pair(scale, shift, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    d_raw = rng.uniform(0.5, 10.0, n).astype(np.float32)
    d_rendered = scale * d_raw + shift
    mask = np.ones(n, dtype=bool)
    return d_raw, d_rendered, mask


def test_recover_known_scale_shift():
    true_scale, true_shift = 2.5, 0.3
    d_raw, d_rendered, mask = _make_aligned_pair(true_scale, true_shift)
    result = align_depth_irls(d_raw, d_rendered, mask)
    assert result.converged
    assert abs(result.scale - true_scale) / true_scale < 0.005
    assert abs(result.shift - true_shift) < 0.02


def test_robust_to_20pct_outliers():
    rng = np.random.default_rng(0)
    true_scale, true_shift = 1.8, 0.1
    d_raw, d_rendered, mask = _make_aligned_pair(true_scale, true_shift, n=2000)
    # Corrupt 20% of rendered values
    n_outliers = 400
    idx = rng.choice(2000, n_outliers, replace=False)
    d_rendered[idx] += rng.uniform(5.0, 20.0, n_outliers)
    result = align_depth_irls(d_raw, d_rendered, mask)
    assert result.converged
    assert abs(result.scale - true_scale) / true_scale < 0.02


def test_scale_only_mode():
    true_scale = 3.0
    d_raw, d_rendered, mask = _make_aligned_pair(true_scale, shift=0.0)
    result = align_depth_irls(d_raw, d_rendered, mask, mode="scale_only")
    assert result.converged
    assert abs(result.scale - true_scale) / true_scale < 0.01
    assert result.shift == 0.0


def test_low_inlier_fraction_marks_not_converged():
    rng = np.random.default_rng(1)
    d_raw = rng.uniform(1.0, 5.0, 100).astype(np.float32)
    d_rendered = rng.uniform(10.0, 50.0, 100).astype(np.float32)  # no relation
    mask = np.ones(100, dtype=bool)
    result = align_depth_irls(d_raw, d_rendered, mask, min_inlier_fraction=0.20)
    # With random unrelated data, residuals will be huge — should not converge
    assert not result.converged or result.inlier_fraction < 0.5
