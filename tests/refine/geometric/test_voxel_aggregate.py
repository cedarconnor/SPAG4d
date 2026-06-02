# tests/refine/geometric/test_voxel_aggregate.py
import numpy as np
import pytest
from spag4d.refine.geometric.aggregate import aggregate_candidates, AggregatedCandidates


def test_aggregate_reduces_point_count():
    # 100 points in a tight cluster -> few voxels
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 0.05, (100, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (100, 3)).astype(np.float32)
    result = aggregate_candidates(pts, colors, voxel_size=0.1)
    assert result.num_voxels < 100
    assert result.positions.shape[1] == 3
    assert result.colors.shape[1] == 3


def test_aggregate_deterministic():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, (500, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (500, 3)).astype(np.float32)
    r1 = aggregate_candidates(pts, colors, voxel_size=0.1)
    r2 = aggregate_candidates(pts, colors, voxel_size=0.1)
    np.testing.assert_array_equal(r1.positions, r2.positions)


def test_aggregate_respects_max_gaussians():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 10, (10000, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (10000, 3)).astype(np.float32)
    result = aggregate_candidates(pts, colors, voxel_size=0.01, max_gaussians=100)
    assert result.num_voxels <= 100
