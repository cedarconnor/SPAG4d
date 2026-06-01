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


def test_sky_mask_excludes_sky_from_depth_range():
    """A learned sky mask should keep sky pixels out of the percentile fit."""
    from spag4d.scene_analysis import compute_scene_defaults

    # Ground 2-8m on the bottom half; "sky" reads as 500m on the top half.
    depth = np.empty((100, 200), dtype=np.float32)
    depth[:50] = 500.0
    rng = np.random.default_rng(0)
    depth[50:] = rng.uniform(2.0, 8.0, size=(50, 200)).astype(np.float32)
    sky = np.zeros((100, 200), dtype=bool)
    sky[:50] = True

    masked = compute_scene_defaults(depth, sky_mask=sky)
    unmasked = compute_scene_defaults(depth)  # sky drags the range up

    assert masked["depth_max"] < 20.0, "sky-excluded depth_max should track the ground"
    assert unmasked["depth_max"] > masked["depth_max"]


def test_sky_mask_none_is_backward_compatible():
    """sky_mask=None must reproduce the original behavior exactly."""
    from spag4d.scene_analysis import compute_scene_defaults

    rng = np.random.default_rng(1)
    depth = rng.uniform(0.5, 5.0, size=(128, 256)).astype(np.float32)
    assert compute_scene_defaults(depth) == compute_scene_defaults(depth, sky_mask=None)
