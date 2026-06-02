# tests/refine/geometric/test_gaussian_init.py
import numpy as np
import pytest
from spag4d.refine.geometric.aggregate import AggregatedCandidates
from spag4d.refine.geometric.init_gaussians import (
    initialize_hole_gaussians,
    new_gaussian_dict,
)


def _mock_aggregated(n=50):
    rng = np.random.default_rng(0)
    return AggregatedCandidates(
        positions=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        colors=rng.uniform(0, 1, (n, 3)).astype(np.float32),
        voxel_size=0.1,
    )


def test_new_gaussian_dict_has_required_keys():
    agg = _mock_aggregated(10)
    d = new_gaussian_dict(agg, knn_scale=0.75)
    for key in ["means", "scales", "quats", "colors", "opacities"]:
        assert key in d, f"Missing key: {key}"


def test_scales_are_positive():
    agg = _mock_aggregated(50)
    d = new_gaussian_dict(agg, knn_scale=0.75)
    assert (d["scales"] > 0).all()


def test_opacities_are_in_logit_range():
    agg = _mock_aggregated(50)
    d = new_gaussian_dict(agg, knn_scale=0.75)
    # logit(0.35) ≈ -0.619
    import scipy.special
    opacities = scipy.special.expit(d["opacities"])
    np.testing.assert_allclose(opacities, 0.35, atol=0.01)


def test_quaternions_are_unit():
    agg = _mock_aggregated(20)
    d = new_gaussian_dict(agg, knn_scale=0.75)
    norms = np.linalg.norm(d["quats"], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
