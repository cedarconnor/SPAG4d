"""Shadow Gaussian creation from synthesized views + aligned depth."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..camera.pinhole import PinholeCamera
from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource
from .depth_estimator import AlignedDepth


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
    valid = conf >= min_confidence
    ys, xs = ys[valid], xs[valid]

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

    # Unproject to 3D using OpenGL camera convention
    # OpenGL: +X right, +Y up, -Z forward
    # Pixel (x, y) → camera-space ray with -Z forward
    dirs_cam = np.stack([
        (xs - camera.cx) / camera.fx,
        -(ys - camera.cy) / camera.fy,  # flip Y (pixel Y-down → OpenGL Y-up)
        -np.ones(N),                      # -Z forward (OpenGL convention)
    ], axis=-1)

    # Normalize
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam = dirs_cam / norms

    # Scale by depth (depth is Z-depth along camera forward axis)
    pts_cam = dirs_cam * depth[:, np.newaxis]

    # Camera → world
    c2w = camera.c2w
    pts_world = (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T

    # Colors from synthesized image
    colors = synthesized_rgb[ys, xs].astype(np.float32)

    # Scales: depth-proportional (similar to SPAG converter)
    angular_spacing = math.pi / H
    base_scale = depth * angular_spacing * stride
    scale_xy = base_scale * 0.5  # Conservative
    scale_z = base_scale * 0.1   # Thin disc
    scales = np.stack([scale_xy, scale_xy, scale_z], axis=-1).astype(np.float32)

    # Quaternions: identity (XYZW)
    quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (N, 1))

    # Opacities: conservative
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
