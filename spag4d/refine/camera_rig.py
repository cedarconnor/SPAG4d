"""Phase 1: Camera placement, rendering, and hole detection."""
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """Perspective camera for novel-view rendering."""
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    fov_deg: float
    width: int
    height: int

    @property
    def intrinsics(self):
        f = self.height / (2 * np.tan(np.radians(self.fov_deg) / 2))
        cx, cy = self.width / 2, self.height / 2
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def generate_camera_rig(
    origin: np.ndarray,
    depth_map: np.ndarray,
    num_directions: int = 12,
    num_depths: int = 3,
    fov_deg: float = 60.0,
    translation_fracs: tuple = (0.05, 0.15, 0.30),
    resolution: int = 512,
) -> list:
    """Generate novel-view cameras that expose disocclusion holes."""
    logger.info(f"[stub] generate_camera_rig: {num_directions} dirs x {num_depths} depths")
    cameras = []
    median_depth = float(np.median(depth_map[depth_map > 0]))
    for azi_idx in range(num_directions):
        azimuth = (2 * np.pi * azi_idx) / num_directions
        for frac in translation_fracs:
            t = frac * median_depth
            cam_pos = origin + t * np.array([np.cos(azimuth), 0.0, np.sin(azimuth)])
            cam = CameraPose(
                position=cam_pos, look_at=origin.copy(),
                up=np.array([0.0, 1.0, 0.0]),
                fov_deg=fov_deg, width=resolution, height=resolution,
            )
            cameras.append(cam)
    return cameras


def render_with_hole_mask(gaussians, camera, alpha_threshold=0.1):
    """Render a camera view and extract hole mask. STUB: returns synthetic data."""
    logger.info("[stub] render_with_hole_mask")
    h, w = camera.height, camera.width
    rgb = np.random.rand(h, w, 3).astype(np.float32) * 0.5 + 0.25
    hole_mask = np.zeros((h, w), dtype=np.float32)
    hole_mask[:64, :64] = 1.0
    hole_mask[-64:, -64:] = 1.0
    return rgb, hole_mask


def select_repair_cameras(cameras, hole_masks, min_hole_fraction=0.03, max_cameras=20):
    """Filter to cameras with significant hole coverage."""
    scored = []
    for i, mask in enumerate(hole_masks):
        frac = float(mask.mean())
        if frac >= min_hole_fraction:
            scored.append((i, frac))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:max_cameras]]


def extract_cubemap_views(panorama, depth_map, face_size=512):
    """Extract 6 cubemap face images and cameras from panorama. STUB."""
    logger.info("[stub] extract_cubemap_views")
    faces = []
    cameras = []
    directions = [
        ([0, 0, -1], [0, 1, 0]), ([0, 0, 1], [0, 1, 0]),
        ([1, 0, 0], [0, 1, 0]), ([-1, 0, 0], [0, 1, 0]),
        ([0, 1, 0], [0, 0, 1]), ([0, -1, 0], [0, 0, -1]),
    ]
    for look_dir, up_dir in directions:
        face = np.random.rand(face_size, face_size, 3).astype(np.float32)
        faces.append(face)
        cameras.append(CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array(look_dir, dtype=np.float64),
            up=np.array(up_dir, dtype=np.float64),
            fov_deg=90.0, width=face_size, height=face_size,
        ))
    return faces, cameras
