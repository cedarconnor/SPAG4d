# spag4d/refine/geometric/erp_unproject.py
"""ERP depth unprojection into world-space 3D points."""
import numpy as np
from .depth_convention import erp_pixel_to_ray


def unproject_erp_depth_to_points(
    depth_z: np.ndarray,
    pose: np.ndarray,
    min_depth: float = 0.01,
) -> np.ndarray:
    """Unproject an ERP z-depth map to world-space 3D points.

    Args:
        depth_z: (H, W) camera-forward z-depth. Zero/negative pixels are skipped.
        pose: (4, 4) camera-to-world transform.
        min_depth: pixels with depth < min_depth are skipped.

    Returns:
        (K, 3) float32 world-space points, K = number of valid pixels.
    """
    H, W = depth_z.shape
    v_idx, u_idx = np.where(depth_z > min_depth)
    if len(u_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_z[v_idx, u_idx].astype(np.float32)
    rays = erp_pixel_to_ray(u_idx, v_idx, H, W)  # (K, 3) unit vectors in cam space

    # Scale rays by z / ray_z_component to get camera-space points
    # ray_z is the z-component of the unit ray
    ray_z = rays[:, 2]
    scale = z / (ray_z + 1e-8)
    cam_pts = rays * scale[:, np.newaxis]  # (K, 3)

    # Transform to world space
    R = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32)
    world_pts = cam_pts @ R.T + t
    return world_pts.astype(np.float32)
