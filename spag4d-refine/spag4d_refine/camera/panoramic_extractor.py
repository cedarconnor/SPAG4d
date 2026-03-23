"""ERP → pinhole extraction via gnomonic projection."""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
from scipy.ndimage import map_coordinates

from .pinhole import PinholeCamera


@dataclass
class PanoExtractionResult:
    """Result of extracting a pinhole view from a panorama."""
    rgb: np.ndarray       # [H, W, 3] float32 [0, 1]
    depth: np.ndarray     # [H, W] float32 Z-depth
    valid_mask: np.ndarray  # [H, W] bool


def extract_panoramic_view(
    panorama_rgb: np.ndarray,
    panorama_depth: np.ndarray,
    camera: PinholeCamera,
    interp_order: int = 1,
) -> PanoExtractionResult:
    """
    Extract a pinhole view from an ERP panorama using gnomonic projection.

    This samples the panorama as if viewed from the panorama center,
    projected onto the novel camera's image plane. No parallax — just
    a perspective crop of the panorama content.

    Args:
        panorama_rgb: [H_pano, W_pano, 3] uint8 or float32
        panorama_depth: [H_pano, W_pano] float32 radial depth
        camera: Target pinhole camera
        interp_order: Interpolation order (0=nearest, 1=bilinear)

    Returns:
        PanoExtractionResult with RGB, Z-depth, and validity mask
    """
    H_pano, W_pano = panorama_depth.shape
    H, W = camera.height, camera.width

    # Ensure float RGB
    if panorama_rgb.dtype == np.uint8:
        rgb_float = panorama_rgb.astype(np.float32) / 255.0
    else:
        rgb_float = panorama_rgb.astype(np.float32)

    # Build pixel grid in camera space
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Pixel → camera-space ray directions (OpenGL: +X right, +Y up, -Z forward)
    dirs_cam = np.stack([
        (uu - camera.cx) / camera.fx,
        -(vv - camera.cy) / camera.fy,  # flip Y (pixel Y-down → OpenGL Y-up)
        -np.ones_like(uu),               # -Z forward (OpenGL convention)
    ], axis=-1)

    # Camera-space → world-space directions
    R = camera.c2w[:3, :3]
    dirs_world = dirs_cam @ R.T  # [H, W, 3]

    # Normalize to unit sphere
    norms = np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    dirs_world = dirs_world / np.clip(norms, 1e-8, None)

    x, y, z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]

    # World direction → ERP pixel coordinates
    # theta (azimuth): atan2(-z, x), mapped to [0, 2π]
    theta = np.arctan2(-z, x)
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # phi (elevation): acos(y), [0, π]
    phi = np.arccos(np.clip(y, -1.0, 1.0))

    # ERP pixel coords
    u_erp = (1.0 - theta / (2 * np.pi)) * (W_pano - 1)
    v_erp = (phi / np.pi) * (H_pano - 1)

    # Sample RGB channels
    out_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        out_rgb[..., c] = map_coordinates(
            rgb_float[..., c], [v_erp, u_erp],
            order=interp_order, mode="wrap",
        )

    # Sample depth (radial)
    radial_depth = map_coordinates(
        panorama_depth, [v_erp, u_erp],
        order=interp_order, mode="wrap",
    ).astype(np.float32)

    # Convert radial depth to Z-depth:
    # Z = radial * cos(angle_from_camera_axis)
    # The angle is between the ray direction and the camera forward (-Z in OpenGL)
    forward = -camera.c2w[:3, 2]  # camera forward in world
    cos_angle = (dirs_world * forward).sum(axis=-1)
    z_depth = radial_depth * np.abs(cos_angle)

    valid = (radial_depth > 0.01) & (radial_depth < 1e4)

    return PanoExtractionResult(
        rgb=out_rgb,
        depth=z_depth,
        valid_mask=valid,
    )
