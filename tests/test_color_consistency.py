import numpy as np
import pytest
import sys
sys.path.insert(0, "spag4d-refine")


def test_consistent_colors_promoted():
    """Gaussians seen with consistent colors across views should be promoted."""
    from spag4d_refine.seeding.shadow_validator import validate_shadow_gaussians
    from spag4d_refine.gaussian.cloud import GaussianCloud
    from spag4d_refine.gaussian.provenance import GaussianSource
    from spag4d_refine.camera.pinhole import PinholeCamera

    cloud = GaussianCloud(
        means=np.array([[0.0, 0.0, -2.0]], dtype=np.float32),
        scales=np.full((1, 3), 0.1, dtype=np.float32),
        quats=np.array([[0, 0, 0, 1]], dtype=np.float32),
        colors=np.array([[0.5, 0.3, 0.2]], dtype=np.float32),
        opacities=np.full((1, 1), 0.85, dtype=np.float32),
        provenance=np.array([GaussianSource.SEEDED], dtype=np.int32),
    )

    cam1 = PinholeCamera.look_at(
        eye=np.array([0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )
    cam2 = PinholeCamera.look_at(
        eye=np.array([-0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )

    synth1 = np.full((64, 64, 3), 0.5, dtype=np.float32)
    synth2 = np.full((64, 64, 3), 0.52, dtype=np.float32)

    result = validate_shadow_gaussians(
        cloud, [cam1, cam2],
        synthesized_images=[synth1, synth2],
        color_consistency_threshold=0.15,
        consistency_threshold=0.3,
    )

    assert result.provenance[0] == GaussianSource.PROMOTED
    assert result.opacities[0] > 0.5


def test_inconsistent_colors_reduced_opacity():
    """Gaussians with conflicting colors across views get reduced opacity."""
    from spag4d_refine.seeding.shadow_validator import validate_shadow_gaussians
    from spag4d_refine.gaussian.cloud import GaussianCloud
    from spag4d_refine.gaussian.provenance import GaussianSource
    from spag4d_refine.camera.pinhole import PinholeCamera

    cloud = GaussianCloud(
        means=np.array([[0.0, 0.0, -2.0]], dtype=np.float32),
        scales=np.full((1, 3), 0.1, dtype=np.float32),
        quats=np.array([[0, 0, 0, 1]], dtype=np.float32),
        colors=np.array([[0.5, 0.3, 0.2]], dtype=np.float32),
        opacities=np.full((1, 1), 0.85, dtype=np.float32),
        provenance=np.array([GaussianSource.SEEDED], dtype=np.int32),
    )

    cam1 = PinholeCamera.look_at(
        eye=np.array([0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )
    cam2 = PinholeCamera.look_at(
        eye=np.array([-0.5, 0.0, 0.0]), target=np.array([0.0, 0.0, -2.0]),
        up=np.array([0, 1, 0]), vfov_deg=60.0, width=64, height=64,
    )

    synth1 = np.full((64, 64, 3), 0.9, dtype=np.float32)
    synth2 = np.full((64, 64, 3), 0.1, dtype=np.float32)

    result = validate_shadow_gaussians(
        cloud, [cam1, cam2],
        synthesized_images=[synth1, synth2],
        color_consistency_threshold=0.15,
        consistency_threshold=0.3,
    )

    assert result.opacities[0] < 0.5, "Opacity should be reduced for color-inconsistent Gaussian"
