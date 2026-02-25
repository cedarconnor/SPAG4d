# spag4d/gaussian_params.py
"""
Per-Gaussian parameterization utilities — numpy reference implementation.

These functions mirror the torch operations in gaussian_converter.py and serve as:
  - Standalone testable implementations (used by tests/test_gaussian_params.py)
  - CPU-only reference / batch-processing helpers
  - Source of truth for the math

Coordinate system (Y-up, right-handed, matches spherical_grid.py):
  rhat  = [sin(phi)*cos(theta),  cos(phi),  -sin(phi)*sin(theta)]
  phi:   colatitude  0 = north pole → pi = south pole
  theta: azimuth     0 .. 2*pi (decreases left-to-right in the ERP image)
"""

import numpy as np
from typing import Tuple


def estimate_normals_from_erp_depth(
    depth_map: np.ndarray,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate surface normals from an ERP depth map using finite differences.

    Converts the depth map to 3D positions, then computes tangent vectors via
    central finite differences in the (phi, theta) angular directions.
    The cross product of the two tangents gives a surface normal at each pixel.

    Args:
        depth_map: (h, w) depth in metres. h and w may differ from H, W when
                   the depth map is at a strided resolution.
        H: original image height (determines angular pixel spacing)
        W: original image width

    Returns:
        normals:    (h, w, 3) unit surface normals pointing **inward** (toward camera)
        confidence: (h, w) values in [0, 1]; low at depth discontinuities
    """
    h, w = depth_map.shape

    # ── Angular coordinate grids at the depth_map resolution ─────────────────
    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Map pixel indices to original-image angular coordinates
    # (depth_map may be at stride resolution, so scale row/col accordingly)
    scale_r = H / h if h != H else 1.0
    scale_c = W / w if w != W else 1.0

    phi   = row_grid * scale_r / H * np.pi                    # colatitude  [0, pi]
    theta = (1.0 - col_grid * scale_c / W) * 2.0 * np.pi     # azimuth     [0, 2pi]

    sin_phi   = np.sin(phi)
    cos_phi   = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # ── 3D positions: P = depth * rhat  (Y-up frame) ─────────────────────────
    x =  depth_map * sin_phi * cos_theta
    y =  depth_map * cos_phi
    z = -depth_map * sin_phi * sin_theta    # negative sign matches spherical_grid.py

    # ── Tangent vector: vertical (phi / row direction) ───────────────────────
    # np.gradient uses central differences in the interior, one-sided at edges.
    dxdp = np.gradient(x, axis=0)
    dydp = np.gradient(y, axis=0)
    dzdp = np.gradient(z, axis=0)
    tangent_phi = np.stack([dxdp, dydp, dzdp], axis=-1)   # (h, w, 3)

    # ── Tangent vector: horizontal (theta / col direction) ───────────────────
    # ERP wraps horizontally: column 0 is adjacent to column W-1.
    # Use explicit circular central differences instead of np.gradient.
    x_next = np.roll(x, -1, axis=1)
    x_prev = np.roll(x,  1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    y_prev = np.roll(y,  1, axis=1)
    z_next = np.roll(z, -1, axis=1)
    z_prev = np.roll(z,  1, axis=1)
    tangent_theta = np.stack([
        (x_next - x_prev) * 0.5,
        (y_next - y_prev) * 0.5,
        (z_next - z_prev) * 0.5,
    ], axis=-1)   # (h, w, 3)

    # ── Surface normal = cross(tangent_phi, tangent_theta) ───────────────────
    normals = np.cross(tangent_phi, tangent_theta)              # (h, w, 3)
    normal_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (normal_mag + 1e-8)

    # Orient to point inward (toward camera at origin)
    positions = np.stack([x, y, z], axis=-1)
    dot = np.sum(normals * positions, axis=-1)                  # (h, w)
    normals[dot > 0] *= -1

    # ── Confidence ────────────────────────────────────────────────────────────
    # Exponential decay from depth gradient magnitude.
    # High gradient → depth discontinuity → unreliable normal → low confidence.
    depth_gx = (np.roll(depth_map, -1, axis=1) - np.roll(depth_map, 1, axis=1)) * 0.5
    depth_gy = np.gradient(depth_map, axis=0)
    depth_grad = np.sqrt(depth_gx ** 2 + depth_gy ** 2)
    median_grad = float(np.median(depth_grad)) + 1e-8
    confidence = np.exp(-depth_grad / median_grad)

    return normals.astype(np.float32), confidence.astype(np.float32)


def compute_gaussian_scales(
    depth_map: np.ndarray,
    stride: int,
    H: int,
    W: int,
    base_scale_factor: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-Gaussian scale proportional to depth and angular pixel footprint.

    For an equirectangular image:
      - Horizontal angular resolution: Δθ = 2π / W  (per original pixel)
      - Vertical angular resolution:   Δφ = π  / H  (per original pixel)
      - At colatitude φ:  horizontal footprint = depth * sin(φ) * Δθ * stride
      -                   vertical footprint   = depth * Δφ * stride

    Args:
        depth_map:        (h, w) depth in metres at the strided resolution.
        stride:           downsampling stride used to produce depth_map.
        H:                original image height.
        W:                original image width.
        base_scale_factor: overall scale multiplier (default 1.2).

    Returns:
        scale_iso: (h, w) isotropic scale — geometric mean of scale_h and scale_v
        scale_h:   (h, w) horizontal (azimuth) scale in metres
        scale_v:   (h, w) vertical (elevation) scale in metres
    """
    h, w = depth_map.shape

    dphi   = np.pi       / H * stride   # radians per stride step (vertical)
    dtheta = 2.0 * np.pi / W * stride   # radians per stride step (horizontal)

    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)
    _, row_grid = np.meshgrid(cols, rows)

    # Map row indices back to original-image colatitude
    phi = row_grid * stride / H * np.pi           # (h, w)
    sin_phi = np.maximum(np.sin(phi), 0.01)        # clamp to avoid zero at poles

    scale_h   = base_scale_factor * depth_map * sin_phi * dtheta
    scale_v   = base_scale_factor * depth_map * dphi
    scale_iso = np.sqrt(scale_h * scale_v)

    return (
        scale_iso.astype(np.float32),
        scale_h.astype(np.float32),
        scale_v.astype(np.float32),
    )


def normal_to_covariance(
    normal: np.ndarray,
    scale_h: float,
    scale_v: float,
    thickness_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a surface normal + tangent scales to 3DGS quaternion + log-scales.

    The Gaussian is modelled as a flat disc aligned with the surface:
      - Two principal axes span the tangent plane (scale_h, scale_v)
      - Third axis is along the surface normal (scale_n = thickness_ratio * min(s_h, s_v))

    Args:
        normal:          (3,) unit surface normal vector
        scale_h:         horizontal tangent scale (metres)
        scale_v:         vertical tangent scale (metres)
        thickness_ratio: disc thinness (0.1 → 10% of min tangent scale)

    Returns:
        quat_wxyz:  (4,) quaternion in **WXYZ** order (3DGS PLY convention)
        log_scales: (3,) log-space scales [log(s_h), log(s_v), log(s_n)]
    """
    n = normal.astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-8)

    # Build orthonormal frame via Gram–Schmidt.
    # Choose the world axis least aligned with n to minimise numerical error.
    abs_n = np.abs(n)
    if abs_n[0] <= abs_n[1] and abs_n[0] <= abs_n[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_n[1] <= abs_n[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1) + 1e-8
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-8

    # Rotation matrix whose columns are [t1, t2, n]
    R = np.stack([t1, t2, n], axis=-1)             # (3, 3)

    # Convert R → quaternion using Shepperd's method (numerically stable)
    # Produces XYZW internally; we convert to WXYZ at the end.
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # 3DGS PLY stores quaternion in WXYZ order
    quat_wxyz = np.array([w, x, y, z], dtype=np.float32)
    quat_wxyz /= np.linalg.norm(quat_wxyz) + 1e-8

    # Disc scales
    scale_n = thickness_ratio * min(float(scale_h), float(scale_v))
    scale_n = max(scale_n, 1e-8)
    log_scales = np.log(
        np.array([float(scale_h), float(scale_v), scale_n], dtype=np.float32) + 1e-8
    )

    return quat_wxyz, log_scales
