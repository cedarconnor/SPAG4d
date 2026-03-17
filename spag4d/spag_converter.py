# spag4d/spag_converter.py
"""
SPAG Gaussian converter: depth map + ERP image -> Gaussians.

Converts an equirectangular depth map directly into 3D Gaussian splat
parameters using spherical projection. Each pixel (at stride resolution)
becomes one Gaussian positioned at depth * ray_direction.

Colors come directly from the panorama pixels (sRGB), so no gamma
correction is needed in the PLY export path.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from .spherical_grid import create_spherical_grid, rotation_matrix_to_quaternion
from .scene_filter import filter_gaussian_candidates, SkyMode


@dataclass
class SPAGParams:
    """Parameters for SPAG depth-to-Gaussian conversion."""
    stride: int = 2
    depth_min: float = 0.1
    depth_max: float = 100.0
    default_opacity: float = 0.9
    disc_thickness: float = 0.1       # z-scale relative to xy-scale
    sky_detection: str = "depth"      # "depth", "gradient", "none"
    sky_threshold: float = 80.0       # depth above this -> sky
    pole_thinning: bool = True
    min_density_ratio: float = 0.30


def depth_to_gaussians(
    erp_image: torch.Tensor,
    depth_map: torch.Tensor,
    params: Optional[SPAGParams] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Convert ERP depth map + image into Gaussian splat parameters.

    Args:
        erp_image: [H, W, 3] uint8 or [0,1] float RGB panorama
        depth_map: [H, W] float depth in meters (radial distance)
        params: SPAG conversion parameters
        device: Torch device for output tensors

    Returns:
        Dict with keys: means [N,3], scales [N,3], quats [N,4] (XYZW),
        colors [N,3] (sRGB 0-1), opacities [N,1]
    """
    if params is None:
        params = SPAGParams()

    if device is None:
        device = depth_map.device if isinstance(depth_map, torch.Tensor) else torch.device('cpu')

    H, W = depth_map.shape[:2]
    stride = params.stride

    # Convert inputs to numpy for filtering
    if isinstance(depth_map, torch.Tensor):
        depth_np = depth_map.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_map, dtype=np.float32)

    if isinstance(erp_image, torch.Tensor):
        img_np = erp_image.detach().cpu().numpy()
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = img_np.astype(np.uint8)
    else:
        img_uint8 = np.asarray(erp_image, dtype=np.uint8)

    # ── 1. Filter: sky detection + depth range + pole thinning ──
    keep_mask, sky_mask = filter_gaussian_candidates(
        depth_map=depth_np,
        erp_image=img_uint8,
        stride=stride,
        sky_mode=SkyMode.SKIP,
        pole_thinning=params.pole_thinning,
        depth_min=params.depth_min,
        depth_max=params.depth_max if params.sky_threshold <= 0 else params.sky_threshold,
        sky_detection=params.sky_detection,
        min_density_ratio=params.min_density_ratio,
    )

    # ── 2. Build spherical grid ──
    grid = create_spherical_grid(H, W, device=device, stride=stride)

    # Grid dimensions
    H_s = grid.rhat.shape[0]
    W_s = grid.rhat.shape[1]

    # Convert keep_mask to torch and flatten
    keep_mask_t = torch.from_numpy(keep_mask).to(device)
    # Ensure shapes match (filter may return slightly different dims)
    keep_mask_t = keep_mask_t[:H_s, :W_s]
    valid_idx = keep_mask_t.nonzero(as_tuple=False)  # [N, 2] (row, col)

    if valid_idx.shape[0] == 0:
        return _empty_gaussians(device)

    rows = valid_idx[:, 0]
    cols = valid_idx[:, 1]
    N = rows.shape[0]

    # ── 3. Sample depth at valid grid positions ──
    # Map strided grid indices back to original pixel coordinates
    pixel_rows = rows * stride + stride // 2
    pixel_cols = cols * stride + stride // 2
    pixel_rows = pixel_rows.clamp(0, H - 1)
    pixel_cols = pixel_cols.clamp(0, W - 1)

    depth_sampled = depth_map[pixel_rows.cpu(), pixel_cols.cpu()].to(device)  # [N]

    # ── 4. Positions: p = depth * r̂ ──
    rhat = grid.rhat[rows, cols]  # [N, 3]
    positions = depth_sampled.unsqueeze(-1) * rhat  # [N, 3]

    # ── 5. Colors: direct from panorama (already sRGB) ──
    if isinstance(erp_image, torch.Tensor):
        img_for_sample = erp_image.float()
        if erp_image.dtype == torch.uint8:
            img_for_sample = img_for_sample / 255.0
    else:
        img_for_sample = torch.from_numpy(img_uint8).float() / 255.0

    colors = img_for_sample[pixel_rows.cpu(), pixel_cols.cpu()].to(device)  # [N, 3]

    # ── 6. Scales: latitude-aware anisotropic ──
    # Angular spacing between grid samples
    angular_spacing = math.pi / H  # radians per pixel row
    base_scale = depth_sampled * angular_spacing * stride  # [N]

    # Latitude correction: sin(phi) shrinks scale near poles
    phi = grid.phi[rows, cols]  # [N]
    sin_phi = torch.sin(phi).clamp(min=0.05)
    scale_xy = base_scale * sin_phi

    # Disc shape: thin along normal direction
    scale_z = base_scale * params.disc_thickness

    scales = torch.stack([scale_xy, scale_xy, scale_z], dim=-1)  # [N, 3]

    # ── 7. Rotations: normal-aligned using tangent basis ──
    # Build rotation matrix from [tangent_right, tangent_up, -rhat]
    # These form an orthonormal frame where -rhat points toward camera
    tangent_right = grid.tangent_right[rows, cols]  # [N, 3]
    tangent_up = grid.tangent_up[rows, cols]        # [N, 3]
    normal = -rhat                                   # [N, 3]

    # R = [right | up | normal] as column vectors
    R = torch.stack([tangent_right, tangent_up, normal], dim=-1)  # [N, 3, 3]
    quats = rotation_matrix_to_quaternion(R)  # [N, 4] XYZW

    # ── 8. Opacities: uniform default ──
    opacities = torch.full((N, 1), params.default_opacity, device=device)

    return {
        'means': positions,
        'scales': scales,
        'quats': quats,
        'colors': colors,
        'opacities': opacities,
    }


def _empty_gaussians(device: torch.device) -> dict:
    """Return empty Gaussian dict with correct shapes."""
    return {
        'means': torch.zeros(0, 3, device=device),
        'scales': torch.zeros(0, 3, device=device),
        'quats': torch.zeros(0, 4, device=device),
        'colors': torch.zeros(0, 3, device=device),
        'opacities': torch.zeros(0, 1, device=device),
    }
