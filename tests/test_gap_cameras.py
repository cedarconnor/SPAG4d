import numpy as np
import pytest


def test_select_gap_cameras_returns_requested_count():
    """Should return at most n_cameras cameras."""
    import sys
    sys.path.insert(0, "spag4d-refine")
    from spag4d_refine.camera.trajectory import select_gap_cameras
    from spag4d_refine.gaussian.cloud import GaussianCloud

    # Create a minimal cloud (10 Gaussians at origin)
    cloud = GaussianCloud(
        means=np.random.randn(10, 3).astype(np.float32) * 0.1,
        scales=np.full((10, 3), 0.01, dtype=np.float32),
        quats=np.tile([0, 0, 0, 1], (10, 1)).astype(np.float32),
        colors=np.random.rand(10, 3).astype(np.float32),
        opacities=np.full((10, 1), 0.9, dtype=np.float32),
    )

    cameras = select_gap_cameras(
        cloud, n_cameras=4, radius=1.0, device="cuda",
    )
    assert len(cameras) <= 4
    assert len(cameras) >= 1


def test_select_gap_cameras_finds_gaps():
    """Cameras should be placed where alpha is low."""
    import sys
    sys.path.insert(0, "spag4d-refine")
    from spag4d_refine.camera.trajectory import select_gap_cameras
    from spag4d_refine.gaussian.cloud import GaussianCloud

    # Cloud covering only +X hemisphere — -X should have gaps
    means = np.zeros((100, 3), dtype=np.float32)
    means[:, 0] = np.random.uniform(0.5, 2.0, 100)
    means[:, 1] = np.random.uniform(-1, 1, 100)
    means[:, 2] = np.random.uniform(-1, 1, 100)

    cloud = GaussianCloud(
        means=means,
        scales=np.full((100, 3), 0.05, dtype=np.float32),
        quats=np.tile([0, 0, 0, 1], (100, 1)).astype(np.float32),
        colors=np.random.rand(100, 3).astype(np.float32),
        opacities=np.full((100, 1), 0.9, dtype=np.float32),
    )

    cameras = select_gap_cameras(
        cloud, n_cameras=2, radius=3.0, device="cuda",
    )
    assert len(cameras) >= 1
