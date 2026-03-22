"""Common test fixtures for spag4d-refine."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from spag4d_refine.gaussian.cloud import GaussianCloud
from spag4d_refine.camera.pinhole import PinholeCamera, CameraSet


@pytest.fixture
def sample_cloud():
    """Create a small GaussianCloud for testing."""
    N = 100
    rng = np.random.default_rng(42)
    return GaussianCloud(
        means=rng.standard_normal((N, 3)).astype(np.float32),
        scales=np.abs(rng.standard_normal((N, 3)).astype(np.float32)) * 0.1 + 0.01,
        quats=_random_quats(N, rng),
        colors=rng.random((N, 3)).astype(np.float32),
        opacities=rng.random((N, 1)).astype(np.float32) * 0.8 + 0.1,
    )


@pytest.fixture
def sample_camera():
    """Create a camera at the origin looking along -Z."""
    c2w = np.eye(4, dtype=np.float64)
    return PinholeCamera.from_fov(60.0, 640, 480, c2w)


@pytest.fixture
def sample_panorama():
    """Create a small synthetic panorama + depth."""
    H, W = 128, 256
    rng = np.random.default_rng(42)
    rgb = rng.random((H, W, 3)).astype(np.float32)
    depth = rng.random((H, W)).astype(np.float32) * 10 + 1.0  # 1-11m
    return rgb, depth


@pytest.fixture
def tmp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _random_quats(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random unit quaternions in XYZW order."""
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q
