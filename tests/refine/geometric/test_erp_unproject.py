# tests/refine/geometric/test_erp_unproject.py
import numpy as np
from spag4d.refine.geometric.erp_unproject import unproject_erp_depth_to_points


def test_center_pixel_projects_along_z():
    H, W = 480, 960
    depth = np.zeros((H, W), dtype=np.float32)
    depth[H // 2, W // 2] = 5.0  # only center pixel has depth
    pose = np.eye(4, dtype=np.float32)
    pts = unproject_erp_depth_to_points(depth, pose)
    # Only non-zero depth pixels are returned
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 5.0], atol=1e-4)


def test_pose_translation_applied():
    H, W = 480, 960
    depth = np.zeros((H, W), dtype=np.float32)
    depth[H // 2, W // 2] = 2.0
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]  # camera at (1,2,3) in world
    pts = unproject_erp_depth_to_points(depth, pose)
    np.testing.assert_allclose(pts[0], [1.0, 2.0, 5.0], atol=1e-4)


def test_returns_empty_for_zero_depth():
    H, W = 64, 128
    depth = np.zeros((H, W), dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pts = unproject_erp_depth_to_points(depth, pose)
    assert pts.shape == (0, 3)
