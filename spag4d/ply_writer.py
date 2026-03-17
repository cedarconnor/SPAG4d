# spag4d/ply_writer.py
"""
PLY export for 3D Gaussian Splats.

Output format is compatible with gsplat, SuperSplat, and other 3DGS viewers.
"""

import numpy as np
from plyfile import PlyData, PlyElement
from typing import Optional

# Spherical harmonics DC normalization constant
SH_C0 = 0.28209479177387814


def _linearRGB_to_sRGB(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB OETF, matching Apple Metal Spec 7.7.7."""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.clip(linear, 0.0031308, None) ** (1.0 / 2.4) - 0.055
    )


def save_ply_gsplat(
    gaussians: dict,
    path: str,
    sh_degree: int = 0,
    colors_linear: bool = True,
) -> None:
    """
    Save Gaussians to PLY format compatible with gsplat viewers.

    Args:
        gaussians: Dict with means, scales, quats, colors, opacities,
                   and optionally 'sh1' [N,9] SH band-1 coefficients
        path: Output PLY file path
        sh_degree: SH degree (0 = DC only, 1 = band-1, ...)
                   Auto-elevated to 1 if gaussians contains 'sh1'.
        colors_linear: If True (SHARP path), apply linearRGB->sRGB conversion.
                       If False (SPAG path), colors are already sRGB — skip gamma.
    """
    # Helper to ensure numpy
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        return np.array(x)

    # Move to CPU numpy
    means = to_numpy(gaussians['means'])
    scales = to_numpy(gaussians['scales'])
    quats = to_numpy(gaussians['quats'])      # XYZW order internally
    colors = to_numpy(gaussians['colors'])
    opacities = to_numpy(gaussians['opacities'])
    
    N = means.shape[0]
    if N == 0:
        raise ValueError("No valid Gaussians to save")
    
    # Coordinate system: Y-up (OpenGL convention) — no transform needed.
    means_out = means
    quats_out = quats

    
    # ─────────────────────────────────────────────────────────────────
    # Encode for PLY storage
    # ─────────────────────────────────────────────────────────────────
    
    # Scales: log-space
    log_scales = np.log(np.clip(scales, 1e-7, None))
    
    # Colors -> SH DC coefficients
    # SHARP outputs linearRGB; SPAG outputs sRGB directly from pixels.
    colors_clamped = np.clip(colors, 0.0, 1.0)
    if colors_linear:
        # SHARP path: linearRGB -> sRGB conversion needed
        colors_srgb = _linearRGB_to_sRGB(colors_clamped)
    else:
        # SPAG path: already sRGB, skip gamma
        colors_srgb = colors_clamped
    sh_dc = (colors_srgb - 0.5) / SH_C0
    
    # Opacity: logit-space
    opacities_clamped = np.clip(opacities, 1e-6, 1 - 1e-6)
    opacity_logit = np.log(opacities_clamped / (1 - opacities_clamped))
    
    # ─────────────────────────────────────────────────────────────────
    # Build PLY structure
    # ─────────────────────────────────────────────────────────────────
    
    # Base properties (always present)
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    
    # Add SH rest coefficients if degree > 0
    if sh_degree >= 1:
        # Total SH coefficients: (degree+1)^2 * 3 channels
        # DC takes 3, rest is total - 3
        num_rest = (sh_degree + 1) ** 2 * 3 - 3
        for i in range(num_rest):
            dtype_list.append((f'f_rest_{i}', 'f4'))
    
    dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    data = np.zeros(N, dtype=dtype_list)
    
    # Fill data
    data['x'], data['y'], data['z'] = means_out.T
    data['nx'] = data['ny'] = data['nz'] = 0  # Unused
    data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = sh_dc.T
    
    # Fill SH rest coefficients if present
    if sh_degree >= 1:
        sh1 = gaussians.get('sh1', None)
        if sh1 is not None:
            sh1_np = to_numpy(sh1)  # [N, 9]
            # Write the 9 band-1 coefficients per Gaussian
            # f_rest ordering in 3DGS PLY: channel-first within each SH function
            # f_rest_{0..2} = Y_1^-1 for R,G,B
            # f_rest_{3..5} = Y_1^0  for R,G,B
            # f_rest_{6..8} = Y_1^1  for R,G,B
            for i in range(min(9, sh1_np.shape[1])):
                data[f'f_rest_{i}'] = sh1_np[:, i]
            # Zero out any remaining higher-band rest coefficients
            num_rest = (sh_degree + 1) ** 2 * 3 - 3
            for i in range(9, num_rest):
                data[f'f_rest_{i}'] = 0
        else:
            num_rest = (sh_degree + 1) ** 2 * 3 - 3
            for i in range(num_rest):
                data[f'f_rest_{i}'] = 0
    
    data['opacity'] = opacity_logit.squeeze()
    data['scale_0'], data['scale_1'], data['scale_2'] = log_scales.T
    
    # Quaternion: PLY uses WXYZ order, internal is XYZW
    data['rot_0'] = quats_out[:, 3]  # W
    data['rot_1'] = quats_out[:, 0]  # X
    data['rot_2'] = quats_out[:, 1]  # Y
    data['rot_3'] = quats_out[:, 2]  # Z
    
    # Write
    el = PlyElement.describe(data, 'vertex')
    PlyData([el], text=False).write(path)
    print(f"Saved {N:,} Gaussians to {path} (SH degree {sh_degree})")


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication (XYZW order).
    
    Args:
        q1: First quaternion [4] or [1, 4]
        q2: Second quaternions [..., 4]
    
    Returns:
        Product q1 * q2 [..., 4]
    """
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # X
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # Y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # Z
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # W
    ], axis=-1)


def load_ply_gaussians(path: str) -> dict:
    """
    Load Gaussians from PLY file.
    
    Args:
        path: Path to PLY file
    
    Returns:
        Dict with decoded means, scales, quats, colors, opacities
    """
    import torch
    
    ply = PlyData.read(path)
    vertex = ply['vertex'].data
    
    # Positions (already in OpenCV coords)
    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    
    # Scales from log
    scales = np.exp(np.stack([
        vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
    ], axis=-1))
    
    # Colors from SH DC
    colors = np.stack([
        vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
    ], axis=-1) * SH_C0 + 0.5
    colors = np.clip(colors, 0, 1)
    
    # Opacity from logit
    opacity_logit = vertex['opacity']
    opacities = 1 / (1 + np.exp(-opacity_logit))
    
    # Quaternions: PLY WXYZ → internal XYZW
    quats = np.stack([
        vertex['rot_1'],  # X
        vertex['rot_2'],  # Y
        vertex['rot_3'],  # Z
        vertex['rot_0'],  # W
    ], axis=-1)
    
    return {
        'means': torch.from_numpy(means).float(),
        'scales': torch.from_numpy(scales).float(),
        'quats': torch.from_numpy(quats).float(),
        'colors': torch.from_numpy(colors).float(),
        'opacities': torch.from_numpy(opacities[:, np.newaxis]).float(),
    }
