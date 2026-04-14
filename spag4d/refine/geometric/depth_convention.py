"""Normative depth convention for the geometric refine pipeline.

All depth values are camera-forward z-depth (positive forward, zero at camera).
All comparisons involving depth MUST route through is_nearer_than_rendered.
Direct inequalities of the form `a < b - margin` are banned in pipeline code.
"""
import numpy as np


def is_nearer_than_rendered(
    candidate_z: np.ndarray,
    rendered_z: np.ndarray,
    margin_ratio: float = 0.02,
    local_depth_scale: np.ndarray | float | None = None,
) -> np.ndarray:
    """True where candidate_z sits strictly in front of rendered_z by > margin.

    margin = margin_ratio * (local_depth_scale if provided, else rendered_z).
    """
    scale = local_depth_scale if local_depth_scale is not None else rendered_z
    margin = margin_ratio * scale
    return candidate_z < (rendered_z - margin)


def radial_to_z(
    radial: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Convert ERP radial (ray-length) depth to camera-forward z-depth.

    lat, lon in radians. lat=0 is equator, lat=pi/2 is north pole.
    z = radial * cos(lat) * cos(lon)  — projection onto forward (z) axis.
    Camera forward is +Z, aligned with lon=0, lat=0.
    """
    return radial * np.cos(lat) * np.cos(lon)


def erp_pixel_to_ray(
    u: np.ndarray,
    v: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Convert ERP pixel coordinates to unit direction vectors in camera space.

    u: column index (0..W-1), v: row index (0..H-1).
    Returns (N, 3) float32 array of unit vectors [x, y, z].
    Camera convention: +Z forward, +X right, +Y up.
    lon=0, lat=0 maps to (0, 0, 1).
    """
    lon = (u.astype(np.float32) / W - 0.5) * 2.0 * np.pi      # [-pi, pi]
    lat = (0.5 - v.astype(np.float32) / H) * np.pi              # [pi/2, -pi/2]
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    rays = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return (rays / norms).astype(np.float32)


def assert_is_z_depth(depth: np.ndarray, rendered_z_reference: np.ndarray) -> None:
    """Raises ValueError if depth looks like radial depth instead of z-depth.

    Heuristic: if max(depth) > 1.5 * max(rendered_z_reference) it's likely radial.
    """
    ratio = np.nanmax(depth) / (np.nanmax(rendered_z_reference) + 1e-8)
    if ratio > 1.5:
        raise ValueError(
            f"Depth buffer appears to be in radial form (max ratio {ratio:.2f}). "
            "Convert to z-depth via radial_to_z() before passing to pipeline stages."
        )
