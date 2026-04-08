"""View selector: perspective crop extraction and gap-directed filtering.

Provides helpers for extracting perspective crops from equirectangular panoramas
and filtering candidate views by their gap-repair priority.
"""

import math
from typing import List

import numpy as np
from scipy.ndimage import map_coordinates

from .camera_rig import CameraPose


def extract_perspective_crop(
    erp_frame: np.ndarray,
    yaw_deg: float,
    fov_deg: float = 90.0,
    size: int = 256,
    pitch_deg: float = 0.0,
) -> np.ndarray:
    """Extract a perspective crop from an equirectangular frame.

    Builds a rectilinear perspective ray grid, rotates it by yaw (around world Y)
    and pitch (around camera X), then samples the ERP image using the SPAG
    spherical convention.

    Args:
        erp_frame: (H, W, 3) float32 equirectangular image in [0, 1].
        yaw_deg:   Horizontal rotation in degrees (positive = counter-clockwise
                   when viewed from above, i.e. towards -Z at 0 deg).
        fov_deg:   Horizontal and vertical field of view in degrees. Default 90.
        size:      Output image side length in pixels. Default 256.
        pitch_deg: Vertical tilt in degrees (positive = tilt up). Default 0.

    Returns:
        (size, size, 3) float32 numpy array clipped to [0, 1].
    """
    h, w = erp_frame.shape[:2]

    # Focal length for a square image with the given FOV
    f = size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))

    # Build a grid of pixel coords in camera space.
    # Camera convention: X right, Y down, Z forward.
    # Pixel (col, row) maps to direction (u, v, f) un-normalised.
    cols = np.arange(size, dtype=np.float32) - (size - 1) / 2.0  # X
    rows = np.arange(size, dtype=np.float32) - (size - 1) / 2.0  # Y
    cc, rr = np.meshgrid(cols, rows)  # (size, size)

    # Un-normalised camera-space ray directions: (X, Y, Z)
    rays = np.stack([cc, rr, np.full_like(cc, f)], axis=-1)  # (size, size, 3)

    # Normalise
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays / norms  # (size, size, 3) unit vectors

    # Rotate by pitch (around camera X = world X when yaw=0):
    # R_pitch: tilt upward means Y shrinks, Z grows -> pitch_deg rotates in YZ plane.
    # Positive pitch_deg tilts camera up (look direction gains positive Y component).
    pitch_rad = math.radians(pitch_deg)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    # Rotation around X axis: Y' = Y*cp - Z*sp, Z' = Y*sp + Z*cp
    # (positive pitch tilts forward vector upward: -Y direction gets mixed with Z)
    # Camera Y is "down", so tilting up means decreasing Y and increasing Z forward.
    # We define positive pitch_deg as "look up" => forward vector's Y component decreases.
    rays_x = rays[..., 0]
    rays_y = rays[..., 1] * cp + rays[..., 2] * sp
    rays_z = -rays[..., 1] * sp + rays[..., 2] * cp
    rays = np.stack([rays_x, rays_y, rays_z], axis=-1)

    # Rotate by yaw around world Y axis.
    # yaw_deg is counter-clockwise when viewed from above (+Y).
    # At yaw=0 the camera looks along +Z (SPAG convention: theta=atan2(-Z,X),
    # so +Z maps to theta=atan2(-1,0)=3pi/2 which is the left quadrant —
    # yaw=0 means we look toward the right side of the equirectangular image).
    yaw_rad = math.radians(yaw_deg)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    # Rotation around Y: X' = X*cy + Z*sy, Z' = -X*sy + Z*cy
    new_x = rays[..., 0] * cy + rays[..., 2] * sy
    new_y = rays[..., 1]
    new_z = -rays[..., 0] * sy + rays[..., 2] * cy
    rays = np.stack([new_x, new_y, new_z], axis=-1)  # (size, size, 3)

    # Convert direction vectors to ERP pixel coordinates using SPAG convention:
    #   theta = atan2(-Z, X),  [0, 2pi]
    #   phi   = acos(Y),       [0, pi]
    #   pixel_x = (1 - theta / (2*pi)) * (w - 1)
    #   pixel_y = phi / pi * (h - 1)
    X, Y, Z = rays[..., 0], rays[..., 1], rays[..., 2]

    theta = np.arctan2(-Z, X)           # [-pi, pi]
    theta = theta % (2.0 * np.pi)       # [0, 2pi]
    phi = np.arccos(np.clip(Y, -1.0, 1.0))  # [0, pi]

    px = (1.0 - theta / (2.0 * np.pi)) * (w - 1)  # [0, w-1]
    py = phi / np.pi * (h - 1)                      # [0, h-1]

    # Bilinear sample each channel; mode="wrap" handles the horizontal ERP seam.
    out = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(
            erp_frame[..., c].astype(np.float64),
            [py, px],
            order=1,
            mode="wrap",
        ).astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def compute_perspective_pose(
    translation: np.ndarray,
    yaw_deg: float,
    fov_deg: float = 90.0,
    size: int = 256,
) -> CameraPose:
    """Compute a CameraPose for a perspective view at a given translation and yaw.

    Args:
        translation: (3,) world-space camera position.
        yaw_deg:     Horizontal look direction in degrees (same convention as
                     ``extract_perspective_crop``).
        fov_deg:     Field of view in degrees. Default 90.
        size:        Image resolution (width and height). Default 256.

    Returns:
        CameraPose with position, look_at, up and the supplied FOV/size.
    """
    translation = np.asarray(translation, dtype=np.float64)
    yaw_rad = math.radians(yaw_deg)

    # Look-at point: one unit ahead along the yaw direction.
    # At yaw=0 the camera looks along +Z (consistent with extract_perspective_crop).
    look_direction = np.array([math.sin(yaw_rad), 0.0, math.cos(yaw_rad)], dtype=np.float64)
    look_at = translation + look_direction

    return CameraPose(
        position=translation.copy(),
        look_at=look_at,
        up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        fov_deg=fov_deg,
        width=size,
        height=size,
    )


def filter_views_by_gap(
    views: list,
    min_gap_ratio: float = 0.05,
    max_views: int = 200,
) -> list:
    """Filter and prioritise views by their gap-repair coverage ratio.

    Each element of ``views`` must be a dict (or any object with a
    ``gap_ratio`` attribute / key) carrying a numeric ``gap_ratio`` value
    representing the fraction of the rendered view covered by holes/gaps.

    Args:
        views:          Sequence of view descriptors, each with a ``gap_ratio``
                        field accessible via dict key or attribute.
        min_gap_ratio:  Minimum gap ratio for a view to be included. Views with
                        gap_ratio < min_gap_ratio are discarded.
        max_views:      Maximum number of views to return after filtering.

    Returns:
        Filtered list sorted by gap_ratio descending, capped to max_views.
    """
    def _gap(v):
        if isinstance(v, dict):
            return v["gap_ratio"]
        return v.gap_ratio

    filtered = [v for v in views if _gap(v) >= min_gap_ratio]
    filtered.sort(key=_gap, reverse=True)
    return filtered[:max_views]
