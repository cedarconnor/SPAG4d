"""
Forward-warp the equirectangular panorama into a pinhole camera
using depth. Produces parallax-correct RGB and depth with
real disocclusion holes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .pinhole import PinholeCamera

# ---------------------------------------------------------------------------
# Optional numba import — fall back to pure-numpy loop when unavailable
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _zbuffer_splat_numba(
        py_arr: np.ndarray,   # int32 [N]
        px_arr: np.ndarray,   # int32 [N]
        z_arr: np.ndarray,    # float32 [N]
        rgb_arr: np.ndarray,  # float32 [N, 3]
        warped_rgb: np.ndarray,    # float32 [H, W, 3]  (output, mutated)
        warped_depth: np.ndarray,  # float32 [H, W]     (output, mutated)
        valid_mask: np.ndarray,    # bool    [H, W]     (output, mutated)
    ) -> None:
        """Z-buffer splatting loop accelerated with numba.

        Arrays are already sorted farthest-first so that closer pixels
        naturally overwrite farther ones.
        """
        n = py_arr.shape[0]
        for i in range(n):
            py = py_arr[i]
            px = px_arr[i]
            z = z_arr[i]
            if z < warped_depth[py, px]:
                warped_rgb[py, px, 0] = rgb_arr[i, 0]
                warped_rgb[py, px, 1] = rgb_arr[i, 1]
                warped_rgb[py, px, 2] = rgb_arr[i, 2]
                warped_depth[py, px] = z
                valid_mask[py, px] = True

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _zbuffer_splat_numpy(
    py_arr: np.ndarray,
    px_arr: np.ndarray,
    z_arr: np.ndarray,
    rgb_arr: np.ndarray,
    warped_rgb: np.ndarray,
    warped_depth: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Pure-numpy fallback for Z-buffer splatting.

    Uses the same farthest-first ordering trick: since the arrays are
    pre-sorted with the farthest pixels first, we can use numpy fancy
    indexing which keeps the *last* write for duplicate indices.  Because
    closer pixels come later in the array, the last write is the nearest —
    exactly the Z-buffer semantics we want.
    """
    warped_rgb[py_arr, px_arr] = rgb_arr
    warped_depth[py_arr, px_arr] = z_arr
    valid_mask[py_arr, px_arr] = True


@dataclass
class ForwardWarpResult:
    """Output of forward-warping the panorama into a novel camera."""
    warped_rgb: np.ndarray     # [H, W, 3] float32 [0, 1] — holes at disocclusions
    warped_depth: np.ndarray   # [H, W] float32 — Z-depth at warped pixels, inf at holes
    valid_mask: np.ndarray     # [H, W] bool — True where a panorama pixel landed
    coverage: float            # Fraction of pixels with valid content


def forward_warp_panorama(
    panorama_rgb: np.ndarray,
    panorama_depth: np.ndarray,
    novel_camera: PinholeCamera,
    pano_center: Optional[np.ndarray] = None,
    subsample: int = 1,
) -> ForwardWarpResult:
    """
    Forward-warp panorama into novel camera via 3D reprojection.

    Vectorized implementation with farthest-first Z-buffer splatting.

    Args:
        panorama_rgb: [H_pano, W_pano, 3] uint8 or float32
        panorama_depth: [H_pano, W_pano] radial depth in meters
        novel_camera: Target pinhole camera
        pano_center: World position of panorama origin (default: [0,0,0])
        subsample: Process every Nth panorama pixel (speed vs coverage)
    """
    H_pano, W_pano = panorama_depth.shape
    H, W = novel_camera.height, novel_camera.width

    if pano_center is None:
        pano_center = np.zeros(3, dtype=np.float64)
    else:
        pano_center = np.asarray(pano_center, dtype=np.float64)

    # Ensure float RGB
    if panorama_rgb.dtype == np.uint8:
        rgb = panorama_rgb.astype(np.float32) / 255.0
    else:
        rgb = panorama_rgb.astype(np.float32)

    # Step 1: Unproject panorama pixels to 3D
    v_idx, u_idx = np.meshgrid(
        np.arange(0, H_pano, subsample, dtype=np.float64),
        np.arange(0, W_pano, subsample, dtype=np.float64),
        indexing="ij",
    )

    # ERP → spherical (Y-up, matching SPAG-4D spherical_grid.py)
    theta = (1.0 - u_idx / (W_pano - 1)) * 2 * np.pi  # azimuth [2π, 0]
    phi = v_idx / (H_pano - 1) * np.pi                  # elevation [0, π]

    # Spherical → Cartesian (Y-up)
    sin_phi = np.sin(phi)
    directions = np.stack([
        sin_phi * np.cos(theta),   # X
        np.cos(phi),               # Y (up)
        -sin_phi * np.sin(theta),  # Z
    ], axis=-1)

    # Sample depth and RGB at subsampled positions
    depth_sub = panorama_depth[::subsample, ::subsample].astype(np.float64)
    rgb_sub = rgb[::subsample, ::subsample]

    # 3D world points
    points_3d = pano_center + directions * depth_sub[..., np.newaxis]

    # Step 2: Project into novel camera
    # Convert OpenGL w2c → OpenCV w2c for projection (Z forward, Y down)
    w2c_gl = novel_camera.w2c
    w2c_cv = w2c_gl.copy()
    w2c_cv[1, :] = -w2c_gl[1, :]  # flip Y
    w2c_cv[2, :] = -w2c_gl[2, :]  # flip Z
    pts_flat = points_3d.reshape(-1, 3)  # [N, 3]
    pts_cam = (w2c_cv[:3, :3] @ pts_flat.T + w2c_cv[:3, 3:4]).T  # [N, 3]

    z = pts_cam[:, 2]
    valid = z > 0.01

    u_proj = np.full_like(z, -1.0)
    v_proj = np.full_like(z, -1.0)
    u_proj[valid] = pts_cam[valid, 0] / z[valid] * novel_camera.fx + novel_camera.cx
    v_proj[valid] = pts_cam[valid, 1] / z[valid] * novel_camera.fy + novel_camera.cy

    in_bounds = valid & (u_proj >= 0) & (u_proj < W - 0.5) & (v_proj >= 0) & (v_proj < H - 0.5)

    # Step 3: Z-buffer splatting (vectorized farthest-first)
    warped_rgb = np.zeros((H, W, 3), dtype=np.float32)
    warped_depth = np.full((H, W), np.inf, dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    rgb_flat = rgb_sub.reshape(-1, 3)
    u_int = np.round(u_proj).astype(np.int32)
    v_int = np.round(v_proj).astype(np.int32)

    # Sort by depth descending — farthest first, closer overwrites
    hit_indices = np.where(in_bounds)[0]
    sorted_idx = hit_indices[np.argsort(-z[hit_indices])]

    # Vectorized splatting: process in batches by depth order
    # For correct Z-buffering, we iterate (numpy fancy indexing can't handle
    # duplicate target indices with ordering guarantees)
    py_arr = v_int[sorted_idx]
    px_arr = u_int[sorted_idx]
    z_arr = z[sorted_idx].astype(np.float32)
    rgb_arr = rgb_flat[sorted_idx]

    if _HAS_NUMBA:
        _zbuffer_splat_numba(py_arr, px_arr, z_arr, rgb_arr,
                             warped_rgb, warped_depth, valid_mask)
    else:
        _zbuffer_splat_numpy(py_arr, px_arr, z_arr, rgb_arr,
                             warped_rgb, warped_depth, valid_mask)

    coverage = float(np.sum(valid_mask)) / (H * W)

    return ForwardWarpResult(
        warped_rgb=warped_rgb,
        warped_depth=warped_depth,
        valid_mask=valid_mask,
        coverage=coverage,
    )


def forward_warp_trajectory(
    panorama_rgb: np.ndarray,
    panorama_depth: np.ndarray,
    cameras: List[PinholeCamera],
    pano_center: Optional[np.ndarray] = None,
    subsample: int = 1,
) -> List[ForwardWarpResult]:
    """Forward-warp for all cameras in a trajectory."""
    return [
        forward_warp_panorama(panorama_rgb, panorama_depth, cam, pano_center, subsample)
        for cam in cameras
    ]
