"""Phase 1: Camera placement, rendering, and hole detection."""
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Ensure GSFix3D is importable
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


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


def _camera_to_RT(camera: "CameraPose"):
    """Convert CameraPose to (R, T) for GSFix3D Camera.

    Returns:
        R: (3, 3) world-to-camera rotation matrix (numpy float32).
        T: (3,)   translation vector in the W2C frame (numpy float32).

    GSFix3D / Inria convention: the 4x4 W2C matrix is [[R, T], [0, 1]].
    Camera axes: +X right, +Y down, +Z forward (OpenCV / COLMAP style).
    """
    forward = camera.look_at - camera.position
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, camera.up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Re-orthogonalise up so it is exactly perpendicular
    up = np.cross(right, forward)

    # Rows of R are the camera-frame basis vectors expressed in world coords.
    # OpenCV convention: X=right, Y=down (=-up), Z=forward
    R = np.stack([right, -up, forward], axis=0).astype(np.float32)

    # T = -R @ position  (camera translation in W2C frame)
    T = (-R @ camera.position).astype(np.float32)

    return R, T


def render_with_hole_mask(gaussians, camera, alpha_threshold=0.1):
    """Render a camera view using GSFix3D and detect holes via low-alpha regions.

    Args:
        gaussians: A GaussianModel instance loaded on CUDA.
        camera: A CameraPose describing the viewpoint.
        alpha_threshold: Pixels with RGB L2-norm below this are marked as holes
                         (since background colour is black, low norm means no
                         Gaussian coverage).

    Returns:
        rgb: (H, W, 3) float32 numpy array in [0, 1].
        hole_mask: (H, W) float32 numpy array, 1.0 = hole, 0.0 = covered.
    """
    import torch
    from gs.camera import Camera as GSCamera
    from gs.gaussian_renderer import render as gs_render

    # Convert CameraPose -> GSFix3D Camera
    R, T = _camera_to_RT(camera)
    fov_y_rad = np.radians(camera.fov_deg)
    aspect = camera.width / camera.height
    fov_x_rad = 2 * math.atan(math.tan(fov_y_rad / 2) * aspect)

    gs_cam = GSCamera(
        R=R, T=T,
        FoVx=fov_x_rad, FoVy=fov_y_rad,
        width=camera.width, height=camera.height,
    )

    # Minimal pipeline config expected by gs_render
    class _Pipe:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    bg = torch.zeros(3, device="cuda")

    with torch.no_grad():
        result = gs_render(gs_cam, gaussians, _Pipe(), bg)

    # result["render"] is (3, H, W), values clamped to [0, 1]
    rgb = result["render"].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # Hole detection: background is black (bg=0), so low RGB norm means
    # few/no Gaussians projected there.
    alpha_proxy = np.sqrt((rgb ** 2).sum(axis=2))
    hole_mask = (alpha_proxy < alpha_threshold).astype(np.float32)

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
    """Extract 6 cubemap face images and cameras from an equirectangular panorama.

    Each face is produced by casting rays from the cube-face grid into the ERP
    image and bilinearly sampling colour values.

    Args:
        panorama: (H, W, 3) float32 equirectangular image.
        depth_map: (H, W) float32 depth map (same projection; unused for now
                   but kept in the API for future depth-face extraction).
        face_size: Resolution of each square face in pixels.

    Returns:
        faces:   List of 6 (face_size, face_size, 3) float32 arrays.
        cameras: List of 6 CameraPose objects (position at origin, 90-deg FOV).
    """
    from scipy.ndimage import map_coordinates

    h, w = panorama.shape[:2]
    faces = []
    cameras = []

    # (forward_vec, up_vec) for each cube face
    face_defs = [
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),   # front  (-Z)
        (np.array([0.0, 0.0, 1.0]),  np.array([0.0, 1.0, 0.0])),   # back   (+Z)
        (np.array([1.0, 0.0, 0.0]),  np.array([0.0, 1.0, 0.0])),   # right  (+X)
        (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),   # left   (-X)
        (np.array([0.0, 1.0, 0.0]),  np.array([0.0, 0.0, -1.0])),  # top    (+Y)
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # bottom (-Y)
    ]

    for forward, up in face_defs:
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)  # re-orthogonalise

        # Build per-pixel ray directions on a 90-deg FOV grid.
        # v is top-to-bottom = +up to -up (matching OpenCV camera convention
        # where row 0 looks UP in world space, row N-1 looks DOWN).
        u = np.linspace(-1.0, 1.0, face_size)
        v = np.linspace(1.0, -1.0, face_size)
        uu, vv = np.meshgrid(u, v)  # (face_size, face_size)

        # Each pixel's direction = forward + u*right + v*up (un-normalised)
        dirs = (uu[..., None] * right
                + vv[..., None] * up
                + forward)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # Convert 3D direction -> equirectangular pixel coordinates
        # Must match SPAG's spherical_grid.py convention:
        #   θ = atan2(-Z, X),  mapped to [0, 2π]
        #   φ = acos(Y),       mapped to [0, π]
        #   pixel_x = (1 - θ/(2π)) * (w - 1)
        #   pixel_y = φ/π * (h - 1)
        theta_spag = np.arctan2(-dirs[..., 2], dirs[..., 0])  # [-pi, pi]
        theta_spag = theta_spag % (2 * np.pi)                 # [0, 2pi]
        phi_spag = np.arccos(np.clip(dirs[..., 1], -1.0, 1.0))  # [0, pi]

        px = (1.0 - theta_spag / (2 * np.pi)) * (w - 1)      # [0, w-1]
        py = phi_spag / np.pi * (h - 1)                       # [0, h-1]

        # Bilinear sample each channel (mode='wrap' handles horizontal seam)
        face = np.zeros((face_size, face_size, 3), dtype=np.float32)
        for c in range(3):
            face[..., c] = map_coordinates(
                panorama[..., c].astype(np.float64),
                [py, px],
                order=1,
                mode="wrap",
            ).astype(np.float32)

        faces.append(face)
        cameras.append(CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=forward.astype(np.float64),
            up=up.astype(np.float64),
            fov_deg=90.0,
            width=face_size,
            height=face_size,
        ))

    return faces, cameras
