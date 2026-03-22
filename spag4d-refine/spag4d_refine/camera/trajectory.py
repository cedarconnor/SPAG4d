"""Trajectory generation for camera paths."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .pinhole import CameraSet, PinholeCamera


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    """Catmull-Rom spline interpolation at parameter t in [0, 1]."""
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
    )


def _look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a c2w matrix from eye, target, up (OpenGL convention)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def generate_orbit_trajectory(
    center: np.ndarray = np.zeros(3),
    radius: float = 1.0,
    n_cameras: int = 8,
    vfov_deg: float = 60.0,
    resolution: tuple[int, int] = (1920, 1080),
    elevation_deg: float = 0.0,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> CameraSet:
    """
    Generate cameras in a circular orbit around a center point.

    Args:
        center: World-space center of the orbit.
        radius: Orbit radius.
        n_cameras: Number of cameras.
        vfov_deg: Vertical FOV in degrees.
        resolution: (width, height) of each camera.
        elevation_deg: Elevation angle above the horizontal plane.
        up: World up vector.
    """
    cameras = []
    elev_rad = math.radians(elevation_deg)
    width, height = resolution

    for i in range(n_cameras):
        angle = 2 * math.pi * i / n_cameras
        eye = center + np.array([
            radius * math.cos(angle) * math.cos(elev_rad),
            radius * math.sin(elev_rad),
            radius * math.sin(angle) * math.cos(elev_rad),
        ])
        c2w = _look_at_c2w(eye, center, up)
        cam = PinholeCamera.from_fov(vfov_deg, width, height, c2w)
        cameras.append(cam)

    return CameraSet(cameras)


def generate_path_trajectory(
    keyframes: List[np.ndarray],
    n_cameras: int = 30,
    vfov_deg: float = 60.0,
    resolution: tuple[int, int] = (1920, 1080),
    target: np.ndarray = np.zeros(3),
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
    closed: bool = False,
) -> CameraSet:
    """
    Generate cameras along a Catmull-Rom spline through keyframes.

    Args:
        keyframes: List of 3D positions [np.ndarray shape (3,)].
        n_cameras: Total cameras to generate.
        vfov_deg: Vertical FOV in degrees.
        resolution: (width, height).
        target: Look-at target for all cameras.
        up: World up vector.
        closed: If True, the path loops back to the first keyframe.
    """
    if len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes")

    pts = [np.asarray(k, dtype=np.float64) for k in keyframes]
    width, height = resolution

    if closed:
        pts = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        # Duplicate endpoints for Catmull-Rom boundary
        pts = [pts[0]] + pts + [pts[-1]]

    n_segments = len(pts) - 3
    cameras = []

    for i in range(n_cameras):
        t_global = i / max(n_cameras - 1, 1) * n_segments
        seg = int(min(t_global, n_segments - 1))
        t_local = t_global - seg

        pos = _catmull_rom(pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3], t_local)
        c2w = _look_at_c2w(pos, target, up)
        cam = PinholeCamera.from_fov(vfov_deg, width, height, c2w)
        cameras.append(cam)

    return CameraSet(cameras)
