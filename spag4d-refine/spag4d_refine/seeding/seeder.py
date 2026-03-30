"""Shadow Gaussian creation from synthesized views + aligned depth."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..camera.pinhole import PinholeCamera
from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource
from .depth_estimator import AlignedDepth

logger = logging.getLogger(__name__)


def _billboard_quaternions(normal: np.ndarray, n: int) -> np.ndarray:
    """
    Compute XYZW quaternion that rotates [0, 0, 1] to the given normal direction.

    All N Gaussians get the same orientation (billboard facing camera).
    Returns [N, 4] float32 in XYZW order.
    """
    # Target: rotate Z-axis to `normal`
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))
    normal = normal / norm

    dot = np.dot(z_axis, normal)
    if dot > 0.9999:
        # Already aligned
        return np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))
    if dot < -0.9999:
        # Opposite: 180 rotation around X
        return np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))

    # Rotation axis = cross(z, normal), angle = acos(dot)
    axis = np.cross(z_axis, normal)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1, 1))
    half = angle / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    quat = np.array([xyz[0], xyz[1], xyz[2], w], dtype=np.float32)
    return np.tile(quat, (n, 1))


def seed_shadow_gaussians(
    aligned_depth: AlignedDepth,
    synthesized_rgb: np.ndarray,
    gap_mask: np.ndarray,
    camera: PinholeCamera,
    shadow_opacity: float = 0.2,
    stride: int = 2,
    min_confidence: float = 0.1,
) -> GaussianCloud:
    """
    Create shadow Gaussians by unprojecting gap pixels to 3D.

    Args:
        aligned_depth: Metric-aligned depth with confidence
        synthesized_rgb: [H, W, 3] float32 [0, 1]
        gap_mask: [H, W] bool — True at pixels to seed
        camera: PinholeCamera used for the view
        shadow_opacity: Initial opacity for shadow Gaussians
        stride: Sample every Nth gap pixel
        min_confidence: Skip pixels below this confidence

    Returns:
        GaussianCloud with provenance=SEEDED
    """
    H, W = gap_mask.shape

    # Sample gap pixels at stride
    ys, xs = np.where(gap_mask)
    if stride > 1:
        # Subsample
        idx = np.arange(0, len(ys), stride)
        ys, xs = ys[idx], xs[idx]

    if len(ys) == 0:
        return GaussianCloud(
            means=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            quats=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros((0, 1), dtype=np.float32),
            provenance=np.zeros(0, dtype=np.int32),
        )

    # Filter by confidence
    conf = aligned_depth.confidence[ys, xs]
    n_before = len(ys)
    valid = conf >= min_confidence
    ys, xs = ys[valid], xs[valid]
    logger.info(
        f"Seeding: {gap_mask.sum():,} gap pixels → {n_before:,} sampled (stride={stride}) "
        f"→ {len(ys):,} pass confidence (min={min_confidence}, "
        f"depth scale={aligned_depth.scale:.3f}, offset={aligned_depth.offset:.3f})"
    )

    if len(ys) == 0:
        return GaussianCloud(
            means=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            quats=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros((0, 1), dtype=np.float32),
            provenance=np.zeros(0, dtype=np.int32),
        )

    N = len(ys)
    depth = aligned_depth.depth[ys, xs]

    # Unproject to 3D using Z-depth directly (NOT radial distance).
    # OpenGL convention: +X right, +Y up, -Z forward.
    # depth is Z-depth (distance along camera forward axis), so:
    #   x_cam = (px - cx) / fx * z_depth
    #   y_cam = -(py - cy) / fy * z_depth  (flip Y: pixel Y-down → OpenGL Y-up)
    #   z_cam = -z_depth                    (-Z forward in OpenGL)
    # This avoids the bug where normalizing rays then scaling by Z-depth
    # places edge pixels too close to the camera.
    pts_cam = np.stack([
        (xs - camera.cx) / camera.fx * depth,
        -(ys - camera.cy) / camera.fy * depth,
        -depth,
    ], axis=-1)

    # Camera → world
    c2w = camera.c2w
    pts_world = (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T

    # Colors from synthesized image
    colors = synthesized_rgb[ys, xs].astype(np.float32)

    # Scales: pixel-footprint-proportional
    # Each pixel covers ~(depth / focal_length) world units at that depth
    pixel_size_x = depth / camera.fx * stride
    pixel_size_y = depth / camera.fy * stride
    scale_xy = np.maximum(pixel_size_x, pixel_size_y) * 0.5
    scale_z = scale_xy * 0.2  # Thin disc (flattened along viewing direction)
    scales = np.stack([scale_xy, scale_xy, scale_z], axis=-1).astype(np.float32)

    # Quaternions: billboard facing the camera (orient disc normal toward camera)
    # Camera forward in world space is -c2w[:3, 2] (OpenGL: -Z forward)
    cam_forward = -c2w[:3, 2]  # unit vector from camera into scene
    # We want each Gaussian's local Z to point back toward the camera
    # i.e., the disc normal = -cam_forward (pointing AT camera)
    # Compute rotation from identity Z-axis [0,0,1] to -cam_forward
    quats = _billboard_quaternions(-cam_forward, N)

    # Opacities: fill gaps convincingly (higher than conservative 0.5)
    opacities = np.full((N, 1), shadow_opacity, dtype=np.float32)

    # Provenance
    provenance = np.full(N, GaussianSource.SEEDED, dtype=np.int32)

    return GaussianCloud(
        means=pts_world.astype(np.float32),
        scales=scales,
        quats=quats,
        colors=colors,
        opacities=opacities,
        provenance=provenance,
    )
