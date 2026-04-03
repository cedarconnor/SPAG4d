"""Gap seeding: generate new Gaussians at under-covered ERP regions.

Uses the source panorama depth map to seed Gaussians at positions
identified as gaps (holes) in the current splat render.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_erp_ray_directions(h: int, w: int) -> np.ndarray:
    """Compute unit ray directions for each pixel in an ERP (equirectangular) image.

    SPAG convention:
        theta = (1 - u / (w - 1)) * 2 * pi   (longitude, rightward = decreasing)
        phi   = v / (h - 1) * pi              (colatitude, top=0, bottom=pi)
        x = cos(theta) * sin(phi)
        y = cos(phi)
        z = -sin(theta) * sin(phi)

    Args:
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        Float32 array of shape (H, W, 3) with unit ray direction vectors.
    """
    # u and v are pixel-column and pixel-row indices respectively.
    u = np.arange(w, dtype=np.float32)  # shape (W,)
    v = np.arange(h, dtype=np.float32)  # shape (H,)

    # Broadcast to (H, W) grids.
    u_grid = u[np.newaxis, :]  # (1, W)
    v_grid = v[:, np.newaxis]  # (H, 1)

    theta = (1.0 - u_grid / max(w - 1, 1)) * 2.0 * np.pi  # longitude (H, W)
    phi = v_grid / max(h - 1, 1) * np.pi                   # colatitude (H, W)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    x = (cos_theta * sin_phi).astype(np.float32)       # (H, W)
    y = np.broadcast_to(cos_phi, x.shape).copy().astype(np.float32)  # (H, W)
    z = (-sin_theta * sin_phi).astype(np.float32)     # (H, W)

    rays = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # Normalize to unit vectors. At the poles sin_phi -> 0, so handle gracefully.
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    rays = (rays / norms).astype(np.float32)

    return rays


def seed_gap_gaussians(
    source_depth: np.ndarray,
    gap_mask: np.ndarray,
    stride: int = 4,
    initial_opacity: float = 0.01,
) -> dict:
    """Seed new Gaussians at gap locations using the source panorama depth.

    Subsamples both the depth map and the gap mask by ``stride``, then
    places a Gaussian at each subsampled pixel that is flagged as a gap.
    Positions are computed by back-projecting depth along the ERP ray.

    Args:
        source_depth: Float array of shape (H, W) containing radial depth
            values (ray lengths in metres) for each ERP pixel.
        gap_mask: Boolean array of shape (H, W). True = gap pixel that needs
            additional coverage.
        stride: Subsampling factor applied to both spatial dimensions.
            stride=1 uses every pixel; stride=4 uses every 4th pixel.
        initial_opacity: Opacity value assigned to all seeded Gaussians.
            Should be small so the splat optimiser can grow them naturally.

    Returns:
        Dict with keys:
            "positions"   — Float32 array of shape (N, 3) world-space positions.
            "opacities"   — Float32 array of shape (N,) with ``initial_opacity``.
            "provenance"  — String tag "gap_seed".
    """
    h, w = source_depth.shape[:2]

    # Subsampled slices.
    sub_depth = source_depth[::stride, ::stride]  # (H', W')
    sub_mask = gap_mask[::stride, ::stride]        # (H', W')

    # Build ray directions at the subsampled resolution.
    h_sub, w_sub = sub_depth.shape
    rays = compute_erp_ray_directions(h_sub, w_sub)  # (H', W', 3)

    # Select gap pixels.
    gap_pixels = sub_mask.astype(bool)
    n_gap = int(gap_pixels.sum())

    if n_gap == 0:
        logger.info("seed_gap_gaussians: no gap pixels found — returning empty result")
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "opacities": np.zeros((0,), dtype=np.float32),
            "provenance": "gap_seed",
        }

    depth_vals = sub_depth[gap_pixels].astype(np.float32)  # (N,)
    ray_vals = rays[gap_pixels]                              # (N, 3)

    positions = ray_vals * depth_vals[:, np.newaxis]  # (N, 3)
    opacities = np.full((n_gap,), initial_opacity, dtype=np.float32)

    logger.info(
        f"seed_gap_gaussians: seeded {n_gap} Gaussians "
        f"(stride={stride}, opacity={initial_opacity})"
    )

    return {
        "positions": positions.astype(np.float32),
        "opacities": opacities,
        "provenance": "gap_seed",
    }
