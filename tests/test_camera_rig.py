"""Tests for spag4d.refine.camera_rig — geometry + cubemap extraction.

GPU-dependent tests (render_with_hole_mask) are skipped when CUDA is unavailable.
"""

import numpy as np
import pytest

from spag4d.refine.camera_rig import (
    CameraPose,
    _camera_to_RT,
    extract_cubemap_views,
    generate_camera_rig,
    select_repair_cameras,
)


# ---------------------------------------------------------------------------
# generate_camera_rig
# ---------------------------------------------------------------------------

def test_camera_rig_count():
    depth = np.ones((180, 360), dtype=np.float32) * 5.0
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth,
        num_directions=12,
        num_depths=3,
    )
    assert len(cameras) == 36


def test_cameras_translate_away_from_origin():
    depth = np.ones((180, 360), dtype=np.float32) * 10.0
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth,
        num_directions=4,
        num_depths=2,
    )
    for cam in cameras:
        assert np.linalg.norm(cam.position) > 0.01


def test_cameras_look_at_origin():
    depth = np.ones((180, 360), dtype=np.float32) * 5.0
    cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth,
        num_directions=4,
        num_depths=1,
    )
    for cam in cameras:
        np.testing.assert_array_equal(cam.look_at, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# CameraPose.intrinsics
# ---------------------------------------------------------------------------

def test_intrinsics():
    cam = CameraPose(
        position=np.zeros(3),
        look_at=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=90.0,
        width=512,
        height=512,
    )
    K = cam.intrinsics
    assert K.shape == (3, 3)
    # f = 512 / (2 * tan(45)) = 256
    assert abs(K[0, 0] - 256.0) < 1.0


# ---------------------------------------------------------------------------
# _camera_to_RT
# ---------------------------------------------------------------------------

def test_camera_to_RT_identity():
    """Camera at origin looking along -Z should produce near-identity R."""
    cam = CameraPose(
        position=np.array([0.0, 0.0, 0.0]),
        look_at=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        width=256,
        height=256,
    )
    R, T = _camera_to_RT(cam)

    assert R.shape == (3, 3)
    assert T.shape == (3,)
    # T should be zero (camera at origin)
    np.testing.assert_allclose(T, 0.0, atol=1e-6)
    # R should be orthogonal
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_camera_to_RT_translated():
    """Camera displaced from origin should produce non-zero T."""
    cam = CameraPose(
        position=np.array([1.0, 2.0, 3.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        width=256,
        height=256,
    )
    R, T = _camera_to_RT(cam)

    # R must be orthogonal
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    # T = -R @ position, so R.T @ T = -position => position = -R.T @ T
    recovered_pos = -R.T @ T
    np.testing.assert_allclose(recovered_pos, cam.position, atol=1e-5)


# ---------------------------------------------------------------------------
# select_repair_cameras
# ---------------------------------------------------------------------------

def test_select_repair_cameras():
    cameras = [None] * 5
    masks = [
        np.ones((10, 10)) * 0.5,    # 50% holes -> selected
        np.ones((10, 10)) * 0.01,   # 1% holes  -> below threshold
        np.ones((10, 10)) * 0.1,    # 10% holes -> selected
        np.zeros((10, 10)),          # 0% holes  -> below threshold
        np.ones((10, 10)) * 0.04,   # 4% holes  -> selected
    ]
    selected = select_repair_cameras(cameras, masks, min_hole_fraction=0.03)
    assert 0 in selected
    assert 2 in selected
    assert 4 in selected
    assert 1 not in selected
    assert 3 not in selected


# ---------------------------------------------------------------------------
# extract_cubemap_views
# ---------------------------------------------------------------------------

def test_cubemap_extraction_shapes():
    panorama = np.random.rand(64, 128, 3).astype(np.float32)
    depth = np.ones((64, 128), dtype=np.float32) * 5.0
    faces, cameras = extract_cubemap_views(panorama, depth, face_size=32)
    assert len(faces) == 6
    assert len(cameras) == 6
    assert faces[0].shape == (32, 32, 3)
    assert cameras[0].fov_deg == 90.0


def test_cubemap_cameras_at_origin():
    panorama = np.random.rand(64, 128, 3).astype(np.float32)
    depth = np.ones((64, 128), dtype=np.float32)
    _, cameras = extract_cubemap_views(panorama, depth, face_size=16)
    for cam in cameras:
        np.testing.assert_array_equal(cam.position, [0.0, 0.0, 0.0])
        assert cam.width == 16
        assert cam.height == 16


def test_cubemap_face_values_in_range():
    """Face pixel values should be within the source panorama's value range."""
    panorama = np.random.rand(128, 256, 3).astype(np.float32)
    depth = np.ones((128, 256), dtype=np.float32)
    faces, _ = extract_cubemap_views(panorama, depth, face_size=64)
    for face in faces:
        assert face.min() >= -0.01  # allow tiny float imprecision
        assert face.max() <= 1.01


def test_cubemap_constant_panorama():
    """A uniform-colour panorama should produce uniform-colour faces."""
    color = np.array([0.3, 0.6, 0.9], dtype=np.float32)
    panorama = np.ones((64, 128, 3), dtype=np.float32) * color
    depth = np.ones((64, 128), dtype=np.float32)
    faces, _ = extract_cubemap_views(panorama, depth, face_size=32)
    expected = np.ones((32, 32, 3), dtype=np.float32) * color
    for face in faces:
        np.testing.assert_allclose(face, expected, atol=0.02)
