import numpy as np
import pytest


def test_outdoor_scene_defaults():
    """Outdoor scene: depths 1-100m, median ~15m."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.lognormal(mean=2.7, sigma=1.0, size=(512, 1024))
    depth = np.clip(depth, 0.5, 200.0)
    result = compute_scene_defaults(depth)

    assert result["sky_threshold"] > 50.0, "Outdoor sky cutoff should be large"
    assert result["depth_min"] > 0.0
    assert result["depth_max"] > result["depth_min"]
    assert result["orbit_radius"] > 0.1
    assert result["orbit_radius"] < 20.0
    assert "confidence_decay_pixels" in result


def test_indoor_scene_defaults():
    """Indoor scene: depths 0.5-5m, median ~2m."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.uniform(0.5, 5.0, size=(512, 1024))
    result = compute_scene_defaults(depth)

    assert result["sky_threshold"] < 10.0, "Indoor sky cutoff should be small"
    assert result["orbit_radius"] < 1.0, "Indoor radius should be small"
    assert result["depth_min"] >= 0.01


def test_auto_parameters_are_positive():
    """All computed parameters must be positive."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.uniform(1.0, 50.0, size=(256, 512))
    result = compute_scene_defaults(depth)

    for key, val in result.items():
        assert val > 0, f"{key} must be positive, got {val}"


def test_handles_zero_depth():
    """Depth maps with zeros (sky/invalid) shouldn't crash."""
    from spag4d.scene_analysis import compute_scene_defaults

    depth = np.random.uniform(1.0, 20.0, size=(256, 512))
    depth[:50, :] = 0.0  # Sky region
    result = compute_scene_defaults(depth)

    assert result["sky_threshold"] > 0
    assert result["depth_min"] > 0
