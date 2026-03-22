"""PinholeCamera and CameraSet for novel-view capture."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class PinholeCamera:
    """
    Pinhole camera with intrinsics and extrinsics.

    Conventions:
    - c2w: camera-to-world [4, 4] (OpenGL: +X right, +Y up, -Z forward)
    - w2c: world-to-camera [4, 4] (cached inverse of c2w)
    - Y-up coordinate system matching SPAG-4D
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    c2w: np.ndarray  # [4, 4] camera-to-world

    _w2c: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def w2c(self) -> np.ndarray:
        """World-to-camera matrix (cached inverse of c2w)."""
        if self._w2c is None:
            self._w2c = np.linalg.inv(self.c2w)
        return self._w2c

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @property
    def viewmat(self) -> np.ndarray:
        """World-to-camera matrix for gsplat (same as w2c)."""
        return self.w2c

    @property
    def position(self) -> np.ndarray:
        """Camera position in world space."""
        return self.c2w[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        """Camera forward direction in world space (-Z in camera frame)."""
        return -self.c2w[:3, 2]

    @classmethod
    def from_fov(
        cls,
        vfov_deg: float,
        width: int,
        height: int,
        c2w: np.ndarray,
    ) -> PinholeCamera:
        """
        Create camera from vertical FOV and resolution.

        Args:
            vfov_deg: Vertical field of view in degrees.
            width: Image width in pixels.
            height: Image height in pixels.
            c2w: Camera-to-world transform [4, 4].
        """
        vfov_rad = math.radians(vfov_deg)
        fy = height / (2.0 * math.tan(vfov_rad / 2.0))
        fx = fy  # Square pixels
        cx = width / 2.0
        cy = height / 2.0
        c2w = np.asarray(c2w, dtype=np.float64).copy()
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, c2w=c2w)

    @classmethod
    def look_at(
        cls,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = np.array([0.0, 1.0, 0.0]),
        vfov_deg: float = 60.0,
        width: int = 1920,
        height: int = 1080,
    ) -> PinholeCamera:
        """Create a camera looking from eye toward target."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        true_up = np.cross(right, forward)

        # OpenGL convention: camera looks along -Z
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = true_up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = eye

        return cls.from_fov(vfov_deg, width, height, c2w)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "c2w": self.c2w.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PinholeCamera:
        """Deserialize from dict."""
        return cls(
            fx=d["fx"], fy=d["fy"],
            cx=d["cx"], cy=d["cy"],
            width=d["width"], height=d["height"],
            c2w=np.array(d["c2w"], dtype=np.float64),
        )


class CameraSet:
    """Ordered collection of PinholeCameras."""

    def __init__(self, cameras: Optional[List[PinholeCamera]] = None):
        self.cameras: List[PinholeCamera] = cameras or []

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int) -> PinholeCamera:
        return self.cameras[idx]

    def __iter__(self):
        return iter(self.cameras)

    def append(self, camera: PinholeCamera) -> None:
        self.cameras.append(camera)

    def merge(self, other: CameraSet) -> CameraSet:
        """Merge two camera sets."""
        return CameraSet(self.cameras + other.cameras)

    def save_json(self, path: str | Path) -> None:
        """Save camera set to JSON file."""
        data = {
            "cameras": [cam.to_dict() for cam in self.cameras],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> CameraSet:
        """Load camera set from JSON file."""
        with open(path) as f:
            data = json.load(f)
        cameras = [PinholeCamera.from_dict(d) for d in data["cameras"]]
        return cls(cameras)
