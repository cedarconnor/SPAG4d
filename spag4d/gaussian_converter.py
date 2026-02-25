# spag4d/gaussian_converter.py
"""
Convert equirectangular panorama + depth to 3D Gaussians.

This implements the SPAG (Spherical Panorama to Gaussians) algorithm
with latitude-aware anisotropic scaling.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional

from .spherical_grid import SphericalGrid, rotation_matrix_to_quaternion


def _estimate_normals_torch(
    depth: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
) -> tuple:
    """
    Estimate surface normals from ERP depth using central finite differences.

    Converts depth to 3D positions, computes tangent vectors via finite
    differences in the (phi, theta) angular directions, and returns their
    cross product as the surface normal.  The horizontal direction uses
    circular differences to handle the ERP wrap-around at the image seam.

    Args:
        depth: [H, W] depth in metres (at strided grid resolution)
        phi:   [H, W] colatitude in radians [0, pi]   (from SphericalGrid)
        theta: [H, W] azimuth    in radians [0, 2pi]  (from SphericalGrid)

    Returns:
        normals:    [H, W, 3] inward-pointing unit normals
        confidence: [H, W] in [0, 1]; low at depth discontinuities
    """
    sin_phi   = torch.sin(phi)
    cos_phi   = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # 3D positions (Y-up frame, matching spherical_grid.py)
    x =  depth * sin_phi * cos_theta
    y =  depth * cos_phi
    z = -depth * sin_phi * sin_theta

    # ── Vertical tangent (phi / row direction) — no horizontal wrap ──────────
    # Central differences; edge rows use one-sided (replicate boundary).
    x_up   = torch.cat([x[:1, :],  x[:-1, :]], dim=0)
    x_down = torch.cat([x[1:,  :], x[-1:, :]], dim=0)
    y_up   = torch.cat([y[:1, :],  y[:-1, :]], dim=0)
    y_down = torch.cat([y[1:,  :], y[-1:, :]], dim=0)
    z_up   = torch.cat([z[:1, :],  z[:-1, :]], dim=0)
    z_down = torch.cat([z[1:,  :], z[-1:, :]], dim=0)
    tangent_phi = torch.stack([
        (x_down - x_up) * 0.5,
        (y_down - y_up) * 0.5,
        (z_down - z_up) * 0.5,
    ], dim=-1)  # [H, W, 3]

    # ── Horizontal tangent (theta / col direction) — circular wrap ───────────
    x_left  = torch.roll(x, shifts=1,  dims=1)
    x_right = torch.roll(x, shifts=-1, dims=1)
    y_left  = torch.roll(y, shifts=1,  dims=1)
    y_right = torch.roll(y, shifts=-1, dims=1)
    z_left  = torch.roll(z, shifts=1,  dims=1)
    z_right = torch.roll(z, shifts=-1, dims=1)
    tangent_theta = torch.stack([
        (x_right - x_left) * 0.5,
        (y_right - y_left) * 0.5,
        (z_right - z_left) * 0.5,
    ], dim=-1)  # [H, W, 3]

    # Surface normal = cross(tangent_phi, tangent_theta)
    normals = torch.cross(tangent_phi, tangent_theta, dim=-1)   # [H, W, 3]
    normal_mag = normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / normal_mag

    # Orient inward (toward camera at origin)
    positions = torch.stack([x, y, z], dim=-1)
    dot = (normals * positions).sum(dim=-1, keepdim=True)
    normals = torch.where(dot > 0, -normals, normals)

    # ── Confidence ────────────────────────────────────────────────────────────
    # Low at depth discontinuities where the cross-product normal is unreliable.
    d_left  = torch.roll(depth, shifts=1,  dims=1)
    d_right = torch.roll(depth, shifts=-1, dims=1)
    d_up    = torch.cat([depth[:1, :], depth[:-1, :]], dim=0)
    d_down  = torch.cat([depth[1:, :], depth[-1:, :]], dim=0)
    depth_gx = (d_right - d_left) * 0.5
    depth_gy = (d_down  - d_up)   * 0.5
    depth_grad = torch.sqrt(depth_gx ** 2 + depth_gy ** 2)
    median_grad = depth_grad.median().clamp(min=1e-8)
    confidence = torch.exp(-depth_grad / median_grad)

    return normals, confidence


def equirect_to_gaussians(
    image: torch.Tensor,
    depth: torch.Tensor,
    grid: SphericalGrid,
    scale_factor: float = 1.5,
    thickness_ratio: float = 0.1,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    pole_rows: int = 3,
    default_opacity: float = 0.95,
    validity_mask: Optional[torch.Tensor] = None,
    oriented_gaussians: bool = True,
) -> dict:
    """
    Convert equirectangular panorama with depth to 3D Gaussians.
    
    Args:
        image: RGB image [H, W, 3] uint8 or float [0,1]
        depth: Depth map [H, W] in meters
        grid: Precomputed SphericalGrid
        scale_factor: Gaussian scale multiplier (larger = more overlap)
        thickness_ratio: Radial thickness as fraction of tangent scales
        depth_min: Minimum valid depth
        depth_max: Maximum valid depth
        pole_rows: Rows to exclude at top/bottom poles
        default_opacity: Opacity for all Gaussians
        validity_mask: Optional [H, W] mask from depth model (0-1 or bool)
        oriented_gaussians: When True (default), orient Gaussians along
            depth-derived surface normals instead of the pure spherical
            normal.  This makes flat surfaces (floors, walls) render as
            properly-aligned discs rather than camera-facing billboards.
    
    Returns:
        Dict with:
            means: [N, 3] 3D positions (Y-up frame)
            scales: [N, 3] Gaussian scales (azimuth, elevation, normal)
            quats: [N, 4] Quaternions in XYZW order
            colors: [N, 3] RGB colors [0, 1]
            opacities: [N, 1] Opacity values
    """
    import torch.nn.functional as F

    device = grid.device
    H, W = grid.original_H, grid.original_W
    stride = grid.stride

    # Convert image to float if needed
    if image.dtype == torch.uint8:
        colors = image.float() / 255.0
    else:
        colors = image.clone()

    # Downsample to match grid — anti-aliased to avoid Moiré at stride ≥ 2.
    # Colors: area average; Depth: min-pool so foreground edges are preserved.
    H_grid, W_grid = grid.theta.shape
    if colors.shape[0] != H_grid or colors.shape[1] != W_grid:
        if stride > 1:
            colors = F.avg_pool2d(
                colors.permute(2, 0, 1).unsqueeze(0), stride, stride
            ).squeeze(0).permute(1, 2, 0)
        else:
            colors = colors[::stride, ::stride]

    if depth.shape[0] != H_grid or depth.shape[1] != W_grid:
        if stride > 1:
            # Min-pool: foreground (closest) depth wins within each strided cell
            d4 = -depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            p = -F.max_pool2d(d4, kernel_size=stride, stride=stride)  # [1, 1, H', W']
            depth = p.reshape(p.shape[2], p.shape[3])  # [H', W']
        else:
            depth = depth[::stride, ::stride]

    # Downsample validity_mask to match grid if provided
    if validity_mask is not None:
        if validity_mask.shape[0] != H_grid or validity_mask.shape[1] != W_grid:
            if stride > 1:
                # Any valid pixel in cell → cell is valid (max-pool)
                vm = validity_mask.float().unsqueeze(0).unsqueeze(0)
                p_vm = F.max_pool2d(vm, kernel_size=stride, stride=stride)
                validity_mask = p_vm.reshape(p_vm.shape[2], p_vm.shape[3]) > 0.5
            else:
                validity_mask = validity_mask[::stride, ::stride]
    
    # ─────────────────────────────────────────────────────────────────
    # Validity Mask
    # ─────────────────────────────────────────────────────────────────
    valid_mask = (depth > depth_min) & (depth < depth_max)
    
    # Apply learned validity mask if provided
    if validity_mask is not None:
        # Convert to bool if float (threshold at 0.5)
        if validity_mask.dtype == torch.float32 or validity_mask.dtype == torch.float16:
            valid_mask = valid_mask & (validity_mask > 0.5)
        else:
            valid_mask = valid_mask & validity_mask.bool()
    
    # ─────────────────────────────────────────────────────────────────
    # Rotation Matrix → Quaternion
    # ─────────────────────────────────────────────────────────────────
    if oriented_gaussians:
        # Depth-derived surface normals give proper orientation for floors,
        # walls, and other non-radial surfaces.  At depth discontinuities
        # (where the cross-product normal is noisy), we blend back toward
        # the safe spherical normal using the confidence weight.
        depth_normals, normal_conf = _estimate_normals_torch(
            depth, grid.phi, grid.theta
        )   # [H_grid, W_grid, 3],  [H_grid, W_grid]

        # Blend: confident surface pixels → depth normal,
        #        uncertain (edges, sky) → spherical normal
        spherical_n = -grid.rhat                        # [H_grid, W_grid, 3]
        blend = normal_conf.unsqueeze(-1)               # [H_grid, W_grid, 1]
        blended_n = depth_normals * blend + spherical_n * (1.0 - blend)
        blended_n = blended_n / blended_n.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Build tangent frame via Gram–Schmidt (world Y-up as reference).
        # Near-vertical normals (poles, skyward surfaces) fall back to world Z.
        world_up = torch.tensor(
            [0., 1., 0.], device=device, dtype=depth.dtype
        ).view(1, 1, 3).expand(H_grid, W_grid, 3)
        world_z = torch.tensor(
            [0., 0., 1.], device=device, dtype=depth.dtype
        ).view(1, 1, 3).expand(H_grid, W_grid, 3)

        t1 = torch.cross(blended_n, world_up, dim=-1)
        t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        t1_pole = torch.cross(blended_n, world_z, dim=-1)
        t1_pole = t1_pole / t1_pole.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        near_vertical = (blended_n[..., 1].abs() > 0.99).unsqueeze(-1).expand_as(t1)
        t1 = torch.where(near_vertical, t1_pole, t1)

        t2 = torch.cross(blended_n, t1, dim=-1)

        R = torch.stack([t1, t2, blended_n], dim=-1)   # [H_grid, W_grid, 3, 3]
    else:
        # Original spherical-normal frame (backward-compatible)
        normal_conf = torch.ones(H_grid, W_grid, device=device, dtype=depth.dtype)
        normal = -grid.rhat
        right  = grid.tangent_right
        up     = grid.tangent_up
        R = torch.stack([right, up, normal], dim=-1)    # [H_grid, W_grid, 3, 3]

    # Convert to quaternion
    quats = rotation_matrix_to_quaternion(R)            # [H_grid, W_grid, 4] XYZW

    # Poles: instead of excluding top/bottom rows, merge them into fewer
    # larger Gaussians (ERP grid converges at the poles, so many pixels
    # map to nearly the same 3D direction).
    pole_means_list = []
    pole_scales_list = []
    pole_quats_list = []
    pole_colors_list = []
    pole_opacities_list = []

    if pole_rows > 0:
        # Exclude the pole band from the regular valid_mask
        pole_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        pole_mask[:pole_rows, :] = False
        pole_mask[-pole_rows:, :] = False
        valid_mask = valid_mask & pole_mask

        # Collect pole rows separately — merge every merge_factor pixels along azimuth
        W_grid = valid_mask.shape[1]
        for row_idx in range(pole_rows):
            for top_bottom in (0, 1):  # 0 = north pole, 1 = south pole
                r = row_idx if top_bottom == 0 else (-(row_idx + 1))
                # More merging for rows closer to the actual pole
                merge_factor = max(2, W_grid // (4 * (row_idx + 1)))
                num_merged = W_grid // merge_factor
                if num_merged == 0:
                    continue

                for col_start in range(num_merged):
                    c_s = col_start * merge_factor
                    c_e = c_s + merge_factor
                    # Average depth over the azimuthal window
                    d_window = depth[r, c_s:c_e]
                    if d_window.numel() == 0:
                        continue
                    d_avg = d_window.mean()
                    if d_avg <= depth_min or d_avg >= depth_max:
                        continue

                    # Use center pixel's direction
                    c_mid = c_s + merge_factor // 2
                    mean_pos = d_avg.unsqueeze(-1) * grid.rhat[r, c_mid]
                    # Larger scale: covers merge_factor pixels worth of angle
                    scale_az = scale_factor * d_avg * torch.sin(grid.phi[r, c_mid]).clamp(min=0.01) \
                               * (2 * math.pi / W) * merge_factor
                    scale_el = scale_factor * d_avg * (math.pi / H) * stride
                    scale_n = torch.minimum(scale_az, scale_el) * thickness_ratio
                    sc = torch.stack([scale_az, scale_el, scale_n])
                    qt = quats[r, c_mid]
                    col = colors[r, c_s:c_e].mean(dim=0)

                    dx_p = d_window[1:] - d_window[:-1] if d_window.numel() > 1 else torch.zeros(1, device=device)
                    dg = dx_p.abs().mean() / d_avg.clamp(min=0.1)
                    op = (0.99 - 0.5 * torch.sigmoid(dg * 10 - 3)).clamp(0.01, 0.99).unsqueeze(-1)

                    pole_means_list.append(mean_pos)
                    pole_scales_list.append(sc)
                    pole_quats_list.append(qt)
                    pole_colors_list.append(col)
                    pole_opacities_list.append(op)
    
    # ─────────────────────────────────────────────────────────────────
    # 3D Positions: P = depth * r̂
    # ─────────────────────────────────────────────────────────────────
    means = depth.unsqueeze(-1) * grid.rhat
    
    # ─────────────────────────────────────────────────────────────────
    # Anisotropic Scale (latitude-aware, content-adaptive)
    # ─────────────────────────────────────────────────────────────────
    # Angular extent per pixel
    delta_theta = 2 * math.pi / W  # Horizontal angular width
    delta_phi = math.pi / H        # Vertical angular height

    # sin(φ) compensates for ERP distortion at poles
    sin_phi = torch.sin(grid.phi).clamp(min=0.01)  # Avoid zero at poles

    # Content-adaptive scale: image gradient → smaller Gaussians in textured/edge
    # areas for detail, larger ones in smooth regions for efficiency.
    img_gray = colors.mean(dim=-1)  # [H_grid, W_grid]
    gx = F.pad((img_gray[:, 1:] - img_gray[:, :-1]).abs(), (0, 1))       # pad right col
    gy = F.pad((img_gray[1:, :] - img_gray[:-1, :]).abs(), (0, 0, 0, 1)) # pad bottom row
    img_grad = torch.sqrt(gx ** 2 + gy ** 2)
    # detail_factor ∈ [0.3, 1.0]: high gradient → smaller Gaussians
    detail_factor = (1.0 / (1.0 + img_grad * 5.0)).clamp(0.3, 1.0)

    # Tangent plane footprint at distance d
    s_azimuth   = scale_factor * depth * sin_phi * delta_theta * stride * detail_factor
    s_elevation = scale_factor * depth * delta_phi * stride * detail_factor

    # s_normal: thin in radial direction
    s_normal = torch.minimum(s_azimuth, s_elevation) * thickness_ratio

    # Stack: [azimuth, elevation, normal]
    scales = torch.stack([s_azimuth, s_elevation, s_normal], dim=-1)
    
    # (Quaternion generation moved up to satisfy pole merging logic)
    
    # ─────────────────────────────────────────────────────────────────
    # SH Band-1 — First-order view-dependent color coefficients
    # ─────────────────────────────────────────────────────────────────
    # From a single ERP panorama, the image gradient encodes how color
    # changes with viewing direction. Map ERP gradient → 3D gradient
    # via the spherical-coordinate Jacobian, then divide by SH_C1.
    # SH_C1 = sqrt(3 / (4π)) ≈ 0.4886.  Band-1 basis: y, z, x.
    SH_C1 = 0.4886025119029199
    theta = grid.theta   # [H_grid, W_grid]  azimuth
    phi   = grid.phi     # elevation from equator

    # Image gradient in ERP pixel coordinates
    # gx = dColor/dTheta (horizontal), gy = dColor/dPhi (vertical)
    col_r = colors[..., 0]; col_g = colors[..., 1]; col_b = colors[..., 2]
    def _grad(ch):
        gx_c = F.pad((ch[:, 1:] - ch[:, :-1]), (0, 1))
        gy_c = F.pad((ch[1:, :] - ch[:-1, :]), (0, 0, 0, 1))
        return gx_c, gy_c
    gx_r, gy_r = _grad(col_r)
    gx_g, gy_g = _grad(col_g)
    gx_b, gy_b = _grad(col_b)

    # Jacobian: (dtheta/dx3d, dphi/dx3d) in terms of x,y,z on unit sphere
    #   theta = atan2(z, x),  phi = asin(y)
    # dtheta/dx = -z/(x²+z²),  dtheta/dy = 0,       dtheta/dz = x/(x²+z²)
    # dphi/dx   = -xy/sqrt(...), dphi/dy = sqrt(x²+z²), dphi/dz = -yz/sqrt(...)
    rx = grid.rhat[..., 0]  # x component of unit direction
    ry = grid.rhat[..., 1]  # y
    rz = grid.rhat[..., 2]  # z
    xz2 = (rx ** 2 + rz ** 2).clamp(min=1e-6)
    r2  = (rx ** 2 + ry ** 2 + rz ** 2).clamp(min=1e-6)
    sqrt_xz2 = xz2.sqrt()

    dtheta_dx = -rz / xz2
    dtheta_dz =  rx / xz2
    # dphi relies on y-up convention: phi = asin(y), rhat on unit sphere
    dphi_dx = -rx * ry / (r2 * sqrt_xz2)
    dphi_dy =  sqrt_xz2 / r2
    dphi_dz = -rz * ry / (r2 * sqrt_xz2)

    # dColor/d(3d-direction) for each channel
    def _dc3d(gx_c, gy_c):
        dc_dx = gx_c * dtheta_dx + gy_c * dphi_dx
        dc_dy = gx_c * 0         + gy_c * dphi_dy
        dc_dz = gx_c * dtheta_dz + gy_c * dphi_dz
        return dc_dx, dc_dy, dc_dz
    dcr_dx, dcr_dy, dcr_dz = _dc3d(gx_r, gy_r)
    dcg_dx, dcg_dy, dcg_dz = _dc3d(gx_g, gy_g)
    dcb_dx, dcb_dy, dcb_dz = _dc3d(gx_b, gy_b)

    # SH band-1 coefficients: coeff = dC/d(basis_dir) / SH_C1
    # Ordering matches 3DGS PLY: f_rest_{0..2}=Y_1^-1 (y-axis), {3..5}=Y_1^0 (z), {6..8}=Y_1^1 (x)
    # Clamp to reasonable range to avoid artifacts from large gradients at edges
    valid_mask_flat = valid_mask.flatten()
    def _sh1(dc):
        return (dc / SH_C1).clamp(-5.0, 5.0).flatten()[valid_mask_flat]
    sh1_r_y = _sh1(dcr_dy); sh1_g_y = _sh1(dcg_dy); sh1_b_y = _sh1(dcb_dy)  # Y_1^-1
    sh1_r_z = _sh1(dcr_dz); sh1_g_z = _sh1(dcg_dz); sh1_b_z = _sh1(dcb_dz)  # Y_1^0
    sh1_r_x = _sh1(dcr_dx); sh1_g_x = _sh1(dcg_dx); sh1_b_x = _sh1(dcb_dx)  # Y_1^1

    # Stack [N, 9]: R_y G_y B_y  R_z G_z B_z  R_x G_x B_x
    sh1_flat = torch.stack([
        sh1_r_y, sh1_g_y, sh1_b_y,
        sh1_r_z, sh1_g_z, sh1_b_z,
        sh1_r_x, sh1_g_x, sh1_b_x,
    ], dim=-1)  # [N, 9]

    # ─────────────────────────────────────────────────────────────────
    # Merge pole Gaussians into output
    # ─────────────────────────────────────────────────────────────────
    valid_mask_flat = valid_mask.flatten()
    means_flat = means.reshape(-1, 3)[valid_mask_flat]
    scales_flat = scales.reshape(-1, 3)[valid_mask_flat]
    quats_flat = quats.reshape(-1, 4)[valid_mask_flat]
    colors_flat = colors.reshape(-1, 3)[valid_mask_flat]
    
    # Opacities
    if default_opacity < 0:
        # Use depth gradient to define opacity
        dx = F.pad(depth[:, 1:] - depth[:, :-1], (0, 1))
        dy = F.pad(depth[1:, :] - depth[:-1, :], (0, 0, 0, 1))
        grad_mag = torch.sqrt(dx**2 + dy**2) / depth.clamp(min=0.1)
        # Smooth surface -> high opacity, huge jump -> low opacity
        opms = (0.99 - 0.5 * torch.sigmoid(grad_mag * 10 - 3)).unsqueeze(-1)
        opacities_flat = opms.reshape(-1, 1)[valid_mask_flat]
    else:
        # Modulate by normal confidence: confident surfaces → full opacity,
        # uncertain regions (edges, discontinuities) → slightly reduced.
        # Formula: opacity = clamp(conf * 1.2, 0.01, default_opacity)
        # This fades Gaussians whose orientation is unreliable rather than
        # hiding them entirely — a subtle quality improvement.
        conf_scale = (normal_conf * 1.2).clamp(0.01, 1.0)
        op_map = (conf_scale * default_opacity).clamp(0.01, 0.99)
        opacities_flat = op_map.reshape(-1, 1)[valid_mask_flat]
    if pole_means_list:
        pm = torch.stack(pole_means_list, dim=0)
        ps = torch.stack(pole_scales_list, dim=0)
        pq = torch.stack(pole_quats_list, dim=0)
        pc = torch.stack(pole_colors_list, dim=0)
        po = torch.stack(pole_opacities_list, dim=0)
        pole_sh1 = torch.zeros(pm.shape[0], 9, device=device)

        means_flat     = torch.cat([means_flat,     pm], dim=0)
        scales_flat    = torch.cat([scales_flat,    ps], dim=0)
        quats_flat     = torch.cat([quats_flat,     pq], dim=0)
        colors_flat    = torch.cat([colors_flat,    pc], dim=0)
        opacities_flat = torch.cat([opacities_flat, po], dim=0)
        sh1_flat       = torch.cat([sh1_flat,       pole_sh1], dim=0)

    return {
        'means':     means_flat,
        'scales':    scales_flat,
        'quats':     quats_flat,
        'colors':    colors_flat,
        'opacities': opacities_flat,
        'sh1':       sh1_flat,    # [N, 9] SH band-1 coefficients
    }


def filter_sky(
    gaussians: dict,
    sky_threshold: float = 80.0
) -> dict:
    """
    Remove Gaussians beyond sky threshold distance.
    
    Args:
        gaussians: Gaussian dict from equirect_to_gaussians
        sky_threshold: Max distance in meters
    
    Returns:
        Filtered Gaussian dict
    """
    distances = gaussians['means'].norm(dim=-1)
    valid = distances < sky_threshold
    
    return {
        'means': gaussians['means'][valid],
        'scales': gaussians['scales'][valid],
        'quats': gaussians['quats'][valid],
        'colors': gaussians['colors'][valid],
        'opacities': gaussians['opacities'][valid]
    }


def generate_sky_dome(
    image: torch.Tensor,
    depth: torch.Tensor,
    grid: SphericalGrid,
    sky_threshold: float = 80.0,
    dome_distance: float = 500.0,
    dome_scale: float = 8.0,
    dome_opacity: float = 0.7,
    dome_stride: int = 4,
) -> dict:
    """
    Generate a sky dome of large Gaussians for pixels beyond the sky threshold.
    
    Instead of clipping sky, place large semi-transparent Gaussians at a fixed
    distance to form a continuous backdrop visible from the origin.
    
    Args:
        image: RGB image [H, W, 3] uint8 or float [0,1]
        depth: Depth map [H, W] in meters (full resolution)
        grid: Precomputed SphericalGrid (already strided)
        sky_threshold: Depth beyond which pixels are treated as sky
        dome_distance: Distance (meters) to place sky Gaussians
        dome_scale: Scale multiplier relative to base angular extent
        dome_opacity: Opacity for sky Gaussians (0-1)
        dome_stride: Additional subsampling of sky pixels (sky is smooth)
    
    Returns:
        Gaussian dict (means, scales, quats, colors, opacities)
    """
    device = grid.device
    H, W = grid.original_H, grid.original_W
    stride = grid.stride
    
    # Image & depth to grid resolution
    H_grid, W_grid = grid.theta.shape
    
    if image.dtype == torch.uint8:
        colors = image.float() / 255.0
    else:
        colors = image.clone()
    
    if colors.shape[0] != H_grid or colors.shape[1] != W_grid:
        if stride > 1:
            colors = F.avg_pool2d(
                colors.permute(2, 0, 1).unsqueeze(0),
                kernel_size=stride, stride=stride
            ).squeeze(0).permute(1, 2, 0)
        else:
            colors = colors[::stride, ::stride]

    if depth.shape[0] != H_grid or depth.shape[1] != W_grid:
        if stride > 1:
            d4 = -depth.unsqueeze(0).unsqueeze(0)
            pooled = -F.max_pool2d(d4, kernel_size=stride, stride=stride)
            depth = pooled.reshape(pooled.shape[2], pooled.shape[3])
        else:
            depth = depth[::stride, ::stride]
    
    # Sky mask: pixels beyond threshold
    sky_mask = depth >= sky_threshold
    
    # Additional subsampling for sky (it's smooth, fewer Gaussians needed)
    if dome_stride > 1:
        subsample_mask = torch.zeros_like(sky_mask)
        subsample_mask[::dome_stride, ::dome_stride] = True
        sky_mask = sky_mask & subsample_mask
    
    if sky_mask.sum() == 0:
        # No sky pixels — return empty dict
        return {
            'means': torch.zeros(0, 3, device=device),
            'scales': torch.zeros(0, 3, device=device),
            'quats': torch.zeros(0, 4, device=device),
            'colors': torch.zeros(0, 3, device=device),
            'opacities': torch.zeros(0, 1, device=device),
            'sh1': torch.zeros(0, 9, device=device),
        }
    
    # Place sky Gaussians at fixed distance along the ray direction
    means = dome_distance * grid.rhat
    
    # Compute scales: large Gaussians that overlap to form continuous backdrop
    delta_theta = 2 * math.pi / W
    delta_phi = math.pi / H
    sin_phi = torch.sin(grid.phi).clamp(min=0.01)
    
    # Effective stride for sky = grid stride × dome_stride
    effective_stride = stride * dome_stride
    
    s_azimuth = dome_scale * dome_distance * sin_phi * delta_theta * effective_stride
    s_elevation = torch.full_like(s_azimuth, dome_scale * dome_distance * delta_phi * effective_stride)
    s_normal = torch.minimum(s_azimuth, s_elevation) * 0.01  # Very thin — flat billboard
    
    scales = torch.stack([s_azimuth, s_elevation, s_normal], dim=-1)
    
    # Quaternions: reuse grid's tangent frame
    normal = -grid.rhat
    right = grid.tangent_right
    up = grid.tangent_up
    R = torch.stack([right, up, normal], dim=-1)
    quats = rotation_matrix_to_quaternion(R)
    
    # Flatten and filter to sky pixels
    sky_flat = sky_mask.flatten()
    
    means_flat = means.reshape(-1, 3)[sky_flat]
    scales_flat = scales.reshape(-1, 3)[sky_flat]
    quats_flat = quats.reshape(-1, 4)[sky_flat]
    colors_flat = colors.reshape(-1, 3)[sky_flat]
    
    N = means_flat.shape[0]
    opacities_flat = torch.full((N, 1), dome_opacity, device=device)
    sh1_flat = torch.zeros((N, 9), device=device)
    
    return {
        'means': means_flat,
        'scales': scales_flat,
        'quats': quats_flat,
        'colors': colors_flat,
        'opacities': opacities_flat,
        'sh1': sh1_flat,
    }


def equirect_to_gaussians_refined(
    image: torch.Tensor,
    depth: torch.Tensor,
    grid: SphericalGrid,
    refined_attrs: Optional[object] = None,  # Avoid circular import, pass RefinedAttributes object
    scale_factor: float = 1.5,
    thickness_ratio: float = 0.1,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    pole_rows: int = 3,
    default_opacity: float = 0.95,
    validity_mask: Optional[torch.Tensor] = None,
    scale_blend: float = 0.8,
    opacity_blend: float = 1.0,
    color_blend: float = 0.5,
    oriented_gaussians: bool = True,
) -> dict:
    """
    Convert ERP panorama to Gaussians with optional SHARP refinements.

    When refined_attrs is provided:
    - Opacities are taken from SHARP (blended by opacity_blend)
    - Scales are blended between geometric and SHARP (by scale_blend)
    - Colors can optionally be refined

    Positions and rotations always come from geometric computation
    to maintain 360° consistency.
    """
    # 1. Compute base Gaussians
    base_gaussians = equirect_to_gaussians(
        image, depth, grid,
        scale_factor, thickness_ratio,
        depth_min, depth_max, pole_rows,
        default_opacity, validity_mask,
        oriented_gaussians=oriented_gaussians,
    )

    if refined_attrs is None:
        return base_gaussians

    # 2. Apply SHARP refinements
    # Note: Attributes in base_gaussians are flat [N, ...]
    # We need to sample refined maps at the valid positions

    H_grid, W_grid = grid.theta.shape
    stride = grid.stride

    # Re-compute validity mask — MUST use the same anti-aliased downsampling as
    # equirect_to_gaussians so the number of valid pixels matches exactly.
    depth_downsampled = depth
    if depth.shape[0] != H_grid or depth.shape[1] != W_grid:
        if stride > 1:
            d4 = -depth.unsqueeze(0).unsqueeze(0)
            p = -F.max_pool2d(d4, kernel_size=stride, stride=stride)
            depth_downsampled = p.reshape(p.shape[2], p.shape[3])
        else:
            depth_downsampled = depth[::stride, ::stride]

    valid_mask = (depth_downsampled > depth_min) & (depth_downsampled < depth_max)

    if validity_mask is not None:
        mask_downsampled = validity_mask
        if validity_mask.shape[0] != H_grid or validity_mask.shape[1] != W_grid:
            if stride > 1:
                vm = validity_mask.float().unsqueeze(0).unsqueeze(0)
                p_vm = F.max_pool2d(vm, kernel_size=stride, stride=stride)
                mask_downsampled = p_vm.reshape(p_vm.shape[2], p_vm.shape[3]) > 0.5
            else:
                mask_downsampled = validity_mask[::stride, ::stride]
        if mask_downsampled.dtype in (torch.float32, torch.float16):
            valid_mask = valid_mask & (mask_downsampled > 0.5)
        else:
            valid_mask = valid_mask & mask_downsampled.bool()

    if pole_rows > 0:
        pole_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        pole_mask[:pole_rows, :] = False
        pole_mask[-pole_rows:, :] = False
        valid_mask = valid_mask & pole_mask

    # Flattened indices for sampling SHARP attribute maps
    valid_flat = valid_mask.flatten()

    # Helper to sample and flatten map
    def sample_map(feature_map: torch.Tensor, channels: int) -> torch.Tensor:
        # feature_map: [H, W] or [H, W, C]
        # Resize to grid size if needed
        if feature_map.shape[0] != H_grid or feature_map.shape[1] != W_grid:
            # Add dims for interpolate: [1, C, H, W]
            if feature_map.dim() == 2:
                inp = feature_map.unsqueeze(0).unsqueeze(0)
            else:
                inp = feature_map.permute(2, 0, 1).unsqueeze(0)
                
            out = F.interpolate(inp, size=(H_grid, W_grid), mode='bilinear', align_corners=True)
            
            if feature_map.dim() == 2:
                resized = out.squeeze()
            else:
                resized = out.squeeze(0).permute(1, 2, 0)
        else:
            resized = feature_map
            
        if channels == 1:
             return resized.flatten()[valid_flat].unsqueeze(-1)
        else:
             return resized.reshape(-1, channels)[valid_flat]

    # The base_gaussians might contain extra merged pole Gaussians at the end.
    # ref_*_flat only matches the size of the non-merged valid pixels.
    # So we apply refinement only to the first N_base elements.

    # Refine Opacity
    if refined_attrs.opacities is not None and opacity_blend > 0:
        ref_opacities_flat = sample_map(refined_attrs.opacities, 1)
        N_base = ref_opacities_flat.shape[0]
        # Blend only the non-pole portion
        base_gaussians['opacities'][:N_base] = (
            (1 - opacity_blend) * base_gaussians['opacities'][:N_base] +
            opacity_blend * ref_opacities_flat
        ).clamp(0.01, 0.99)

    # Refine Scales (confidence-weighted: SHARP's opacity output proxies its confidence)
    if refined_attrs.scales is not None and scale_blend > 0:
        ref_scales_flat  = sample_map(refined_attrs.scales, 3)
        ref_opacity_flat = sample_map(refined_attrs.opacities, 1) if refined_attrs.opacities is not None else None

        N_base = ref_scales_flat.shape[0]
        
        # Normalize SHARP scales to get relative variation multiplier
        scale_mult = ref_scales_flat / (ref_scales_flat.mean() + 1e-6)
        scale_mult = scale_mult.clamp(0.5, 2.0)

        if ref_opacity_flat is not None:
            # High SHARP opacity → confident surface → allow stronger SHARP influence
            confidence = ref_opacity_flat.clamp(0.1, 0.9)  # [N_base, 1]
            adaptive_blend = (scale_blend * confidence).clamp(0.0, 1.0)
        else:
            adaptive_blend = scale_blend

        base_gaussians['scales'][:N_base] = (
            (1 - adaptive_blend) * base_gaussians['scales'][:N_base] +
            adaptive_blend * base_gaussians['scales'][:N_base] * scale_mult
        )

    # Refine Colors (blend with source to preserve fidelity)
    if refined_attrs.colors is not None and color_blend > 0:
        ref_colors_flat = sample_map(refined_attrs.colors, 3).clamp(0, 1)
        N_base = ref_colors_flat.shape[0]
        base_gaussians['colors'][:N_base] = (
            (1 - color_blend) * base_gaussians['colors'][:N_base] +
            color_blend * ref_colors_flat
        )

    return base_gaussians
