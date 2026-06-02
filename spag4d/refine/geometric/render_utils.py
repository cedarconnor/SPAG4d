# spag4d/refine/geometric/render_utils.py
"""Cube-face ERP composition for rendering the base splat from arbitrary poses.

Renders six perspective cube faces via gsplat, then reprojects to ERP latlong.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RenderOutput:
    rgb: np.ndarray    # (H, W, 3) float32
    depth: np.ndarray  # (H, W) float32, camera-forward z-depth
    alpha: np.ndarray  # (H, W) float32, accumulated opacity [0, 1]


# Cube face definitions: (face name, yaw_deg, pitch_deg)
# Each row = (yaw_deg, pitch_deg) for the face center direction
_CUBE_FACES = [
    ("front",  0,   0),
    ("back",   180, 0),
    ("right",  90,  0),
    ("left",   -90, 0),
    ("up",     0,   90),
    ("down",   0,  -90),
]


def _face_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return 3x3 rotation matrix for a cube face direction."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    Ry = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    return (Rx @ Ry).astype(np.float32)


def render_base_from_pose(
    gaussians,
    pose: np.ndarray,
    resolution: tuple[int, int],
    near: float = 0.01,
    far: float = 1000.0,
    face_size: int = 512,
) -> RenderOutput:
    """Render base splat from an ERP pose via cube-face composition.

    Args:
        gaussians: GaussianModel loaded via format_compat.
        pose: (4, 4) camera-to-world transform.
        resolution: (H, W) ERP output resolution.
        face_size: pixel width/height of each cube face.

    Returns:
        RenderOutput with rgb, depth, alpha in ERP layout.
    """
    try:
        from gsplat import rasterization
    except ImportError:
        raise ImportError("gsplat is required for geometric refine rendering. "
                          "Install via: pip install gsplat>=1.5.0")

    H_erp, W_erp = resolution
    erp_rgb = np.zeros((H_erp, W_erp, 3), dtype=np.float32)
    erp_depth = np.full((H_erp, W_erp), np.inf, dtype=np.float32)
    erp_alpha = np.zeros((H_erp, W_erp), dtype=np.float32)

    # World-to-camera base transform
    world_to_cam_base = np.linalg.inv(pose).astype(np.float32)

    # Perspective intrinsics for 90-degree FoV cube face
    f = face_size / 2.0
    K = np.array([[f, 0, f], [0, f, f], [0, 0, 1]], dtype=np.float32)

    # Extract Gaussian parameters as tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    means = torch.from_numpy(gaussians.get_xyz.detach().cpu().numpy()).to(device)
    quats = torch.from_numpy(gaussians.get_rotation.detach().cpu().numpy()).to(device)
    scales = torch.exp(torch.from_numpy(gaussians.get_scaling.detach().cpu().numpy())).to(device)
    opacities = torch.sigmoid(torch.from_numpy(gaussians.get_opacity.detach().cpu().numpy()).squeeze(-1)).to(device)
    colors = torch.from_numpy(
        gaussians.get_features.detach().cpu().numpy()[:, 0, :]
    ).to(device)  # SH degree 0 colors, (N, 3)

    for face_name, yaw, pitch in _CUBE_FACES:
        face_R = _face_rotation(yaw, pitch)
        # Compose: world_to_cam_base then face rotation
        face_R4 = np.eye(4, dtype=np.float32)
        face_R4[:3, :3] = face_R
        world_to_face = face_R4 @ world_to_cam_base

        viewmat = torch.from_numpy(world_to_face).unsqueeze(0).to(device)
        K_t = torch.from_numpy(K).unsqueeze(0).to(device)

        renders, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K_t,
            width=face_size,
            height=face_size,
            near_plane=near,
            far_plane=far,
            render_mode="RGB+D",
            sh_degree=0,
        )

        face_rgb = renders[0, :, :, :3].cpu().numpy()   # (F, F, 3)
        face_depth = renders[0, :, :, 3].cpu().numpy()  # (F, F)
        face_alpha = alphas[0, :, :, 0].cpu().numpy()   # (F, F)

        _splat_face_to_erp(
            face_rgb, face_depth, face_alpha,
            yaw, pitch, face_size,
            erp_rgb, erp_depth, erp_alpha,
            H_erp, W_erp,
        )

    erp_depth[np.isinf(erp_depth)] = far
    return RenderOutput(rgb=erp_rgb, depth=erp_depth, alpha=erp_alpha)


def _splat_face_to_erp(
    face_rgb, face_depth, face_alpha,
    yaw_deg, pitch_deg, face_size,
    erp_rgb, erp_depth, erp_alpha,
    H_erp, W_erp,
):
    """Reproject a rendered cube face into the ERP output buffers (nearest-neighbour)."""
    face_R = _face_rotation(yaw_deg, pitch_deg)

    # Build pixel grid for the cube face
    px = np.arange(face_size)
    py = np.arange(face_size)
    pxx, pyy = np.meshgrid(px, py)
    f = face_size / 2.0

    # Face-space ray directions
    rx = (pxx.ravel() - f) / f
    ry = (pyy.ravel() - f) / f
    rz = np.ones(face_size * face_size)
    rays = np.stack([rx, ry, rz], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # Rotate to world (then to ERP camera) using face_R^T (= face_R^{-1})
    world_rays = rays @ face_R  # (N, 3)

    # World ray → ERP lat/lon
    x, y, z = world_rays[:, 0], world_rays[:, 1], world_rays[:, 2]
    lat = np.arcsin(np.clip(y, -1, 1))
    lon = np.arctan2(x, z)

    u_erp = ((lon / (2 * np.pi) + 0.5) * W_erp).astype(np.int32) % W_erp
    v_erp = ((0.5 - lat / np.pi) * H_erp).astype(np.int32)
    v_erp = np.clip(v_erp, 0, H_erp - 1)

    fi = pyy.ravel()
    fj = pxx.ravel()
    d = face_depth[fi, fj]
    valid = d > 0

    # Only overwrite if this face's depth is closer
    existing_depth = erp_depth[v_erp[valid], u_erp[valid]]
    closer = d[valid] < existing_depth

    vi = v_erp[valid][closer]
    ui = u_erp[valid][closer]
    fii = fi[valid][closer]
    fji = fj[valid][closer]

    erp_rgb[vi, ui] = face_rgb[fii, fji]
    erp_depth[vi, ui] = face_depth[fii, fji]
    erp_alpha[vi, ui] = face_alpha[fii, fji]
