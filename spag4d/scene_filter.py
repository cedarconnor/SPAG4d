# spag4d/scene_filter.py
"""
Sky detection and pole-density normalization for SPAG-4D.

Provides masks that control which pixels yield Gaussians:
  - Sky detection: removes garbage geometry at near-infinity depth
  - Pole thinning: normalizes splat density ~uniformly across the sphere

The equirectangular projection oversamples the poles — a pixel at the equator
covers ~1 steradian while a pixel at the pole covers ~0 steradians. Without
correction this produces a dense blob at the top/bottom of every splat scene.
"""

import numpy as np
import torch
from enum import Enum
from typing import Optional, Tuple


class SkyMode(Enum):
    """How to handle pixels detected as sky."""
    SKIP = "skip"                # Drop them entirely (transparent void)
    BACKGROUND_SPHERE = "sphere" # Place on a large background sphere
    LOW_OPACITY = "low_opacity"  # Keep but reduce opacity to 0.15


# ──────────────────────────────────────────────────────────────────────────────
# Sky Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_sky_depth(
    depth_map: np.ndarray,
    depth_max: float = 100.0,
    threshold_ratio: float = 0.90,
) -> np.ndarray:
    """
    Classify pixels as sky using only a depth threshold.

    Fast and model-free. Works well when the depth model assigns near-maximum
    depth to sky regions (as DAP and PanDA do).

    Args:
        depth_map: (H, W) float32 depth in meters
        depth_max: The expected maximum scene depth
        threshold_ratio: Pixels > depth_max * threshold_ratio are sky

    Returns:
        sky_mask: (H, W) bool
    """
    return depth_map > (depth_max * threshold_ratio)


def detect_sky_gradient(
    depth_map: np.ndarray,
    erp_image: np.ndarray,
    depth_max: float = 100.0,
    depth_ratio: float = 0.70,
    grad_percentile: float = 30.0,
    color_var_percentile: float = 20.0,
    patch_size: int = 32,
) -> np.ndarray:
    """
    Classify pixels as sky using three complementary signals (no extra model):

      1. High depth value  (> depth_max * depth_ratio)
      2. Low depth gradient magnitude  (flat depth surface)
      3. Low local color variance  (uniform color patch)

    Args:
        depth_map: (H, W) float32 depth in meters
        erp_image: (H, W, 3) uint8 RGB equirectangular image
        depth_max: Maximum expected scene depth
        depth_ratio: Lower bound multiple of depth_max for sky candidates
        grad_percentile: Gradient percentile threshold (below = low gradient)
        color_var_percentile: Color variance percentile (below = uniform)
        patch_size: Window size for local color variance computation

    Returns:
        sky_mask: (H, W) bool
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        raise ImportError("scipy is required for detect_sky_gradient. Install with: pip install scipy")

    H, W = depth_map.shape

    # Signal 1: High depth
    high_depth = depth_map > (depth_max * depth_ratio)

    # Signal 2: Low depth gradient (sky is geometrically flat)
    dy = np.gradient(depth_map, axis=0)
    dx = np.gradient(depth_map, axis=1)
    grad_mag = np.sqrt(dx**2 + dy**2)
    # Threshold computed only over the high-depth region so the percentile
    # is meaningful (sky has very low gradient; foreground may dominate otherwise)
    if high_depth.any():
        grad_thresh = np.percentile(grad_mag[high_depth], grad_percentile)
    else:
        grad_thresh = np.percentile(grad_mag, grad_percentile)
    low_grad = grad_mag < grad_thresh

    # Signal 3: Low local color variance (sky has uniform color)
    gray = erp_image.mean(axis=-1).astype(np.float32)
    local_mean = uniform_filter(gray, size=patch_size)
    local_sq   = uniform_filter(gray**2, size=patch_size)
    local_var  = np.maximum(local_sq - local_mean**2, 0.0)
    var_thresh = np.percentile(local_var, color_var_percentile)
    low_var = local_var < var_thresh

    # Combine — all three signals must agree
    sky_mask = high_depth & low_grad & low_var
    return sky_mask


# ──────────────────────────────────────────────────────────────────────────────
# Pole Thinning
# ──────────────────────────────────────────────────────────────────────────────

def compute_pole_thinning_mask(
    H: int,
    W: int,
    stride: int,
    min_density_ratio: float = 0.30,
    seed: int = 42,
) -> np.ndarray:
    """
    Stochastic mask that normalises Gaussian density across the sphere.

    At latitude θ the solid angle per pixel is proportional to sin(θ).
    We thin by keeping each pixel with probability sin(θ), floored at
    min_density_ratio so we never completely eliminate pole coverage.

    Args:
        H: Full-resolution image height
        W: Full-resolution image width
        stride: Grid stride (mask is at stride resolution)
        min_density_ratio: Minimum keep probability (prevents zero pole coverage)
        seed: RNG seed for reproducibility

    Returns:
        keep_mask: (n_rows, n_cols) bool at stride resolution
    """
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    n_rows = len(rows)
    n_cols = len(cols)

    # sin(θ): 0 at poles, 1 at equator  (θ = colatitude 0…π)
    theta = rows / H * np.pi
    keep_prob_1d = np.sin(theta).clip(min_density_ratio, 1.0)  # (n_rows,)

    # Broadcast to 2D
    keep_prob_2d = np.tile(keep_prob_1d[:, None], (1, n_cols))  # (n_rows, n_cols)

    rng = np.random.default_rng(seed)
    keep_mask = rng.random(keep_prob_2d.shape) < keep_prob_2d

    return keep_mask


def get_adaptive_row_stride(
    H: int,
    base_stride: int,
    max_stride_factor: int = 4,
) -> np.ndarray:
    """
    Per-row stride that normalises Gaussian density via latitude-adaptive spacing.

    At the equator: stride = base_stride.
    Near the poles:  stride = base_stride * max_stride_factor (sparser grid).

    This is an alternative to stochastic thinning; deterministic and produces
    cleaner patterns at the poles.

    Args:
        H: Full-resolution image height
        base_stride: Stride at the equator (θ = π/2)
        max_stride_factor: Maximum polar stride is base_stride × this value

    Returns:
        row_strides: (H,) int array — effective stride at each row
    """
    rows = np.arange(H)
    theta = rows / H * np.pi  # 0…π colatitude
    sin_theta = np.sin(theta).clip(1.0 / max_stride_factor, 1.0)

    row_strides = np.round(base_stride / sin_theta).astype(int)
    row_strides = np.clip(row_strides, base_stride, base_stride * max_stride_factor)
    return row_strides


# ──────────────────────────────────────────────────────────────────────────────
# Combined Filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_gaussian_candidates(
    depth_map: np.ndarray,
    erp_image: np.ndarray,
    stride: int,
    sky_mode: SkyMode = SkyMode.SKIP,
    pole_thinning: bool = True,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    sky_detection: str = "gradient",  # "gradient" | "depth" | "none"
    min_density_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a boolean mask of valid Gaussian pixel positions.

    Combines:
      - Depth range validation
      - Sky detection & removal
      - Pole-density normalisation

    Args:
        depth_map: (H, W) float32 depth in meters
        erp_image: (H, W, 3) uint8 RGB image  (needed for gradient sky detect)
        stride: Grid stride
        sky_mode: How to handle sky pixels
        pole_thinning: Enable sin(θ) density normalisation
        depth_min: Minimum valid depth
        depth_max: Maximum valid depth
        sky_detection: Algorithm for sky detection
        min_density_ratio: Minimum pole keep probability
        seed: RNG seed for stochastic thinning

    Returns:
        keep_mask: (n_rows, n_cols) bool — True = generate a Gaussian here
        sky_mask_strided: (n_rows, n_cols) bool — True = pixel is sky
    """
    H, W = depth_map.shape

    # Depth at grid positions
    depth_stride = depth_map[::stride, ::stride]

    # ── Depth range ──
    valid_depth = (depth_stride > depth_min) & (depth_stride < depth_max)

    # ── Sky detection ──
    if sky_detection == "gradient":
        sky_mask_full = detect_sky_gradient(depth_map, erp_image, depth_max=depth_max)
    elif sky_detection == "depth":
        sky_mask_full = detect_sky_depth(depth_map, depth_max=depth_max)
    else:
        sky_mask_full = np.zeros((H, W), dtype=bool)

    # Downsample sky mask to stride resolution (point-sampled at grid positions)
    sky_mask_strided = sky_mask_full[::stride, ::stride]

    # ── Pole thinning ──
    if pole_thinning:
        pole_mask = compute_pole_thinning_mask(H, W, stride, min_density_ratio, seed)
    else:
        n_rows = len(np.arange(0, H, stride))
        n_cols = len(np.arange(0, W, stride))
        pole_mask = np.ones((n_rows, n_cols), dtype=bool)

    # ── Combine ──
    if sky_mode == SkyMode.SKIP:
        keep_mask = ~sky_mask_strided & valid_depth & pole_mask
    else:
        # Sky pixels handled separately (background sphere / low opacity pass)
        keep_mask = ~sky_mask_strided & valid_depth & pole_mask

    return keep_mask, sky_mask_strided


def apply_sky_mode_to_gaussians(
    gaussians: dict,
    sky_mask_strided: np.ndarray,
    depth_map_strided: np.ndarray,
    erp_image_strided: np.ndarray,
    sky_mode: SkyMode,
    depth_max: float = 100.0,
    sky_radius: float = 200.0,
    sky_opacity: float = 0.15,
    device: Optional[object] = None,
) -> dict:
    """
    Optionally append sky Gaussians (BACKGROUND_SPHERE or LOW_OPACITY modes).

    For SKIP mode, call filter_gaussian_candidates() with sky_mode=SKIP and
    never call this function.

    Args:
        gaussians: Existing Gaussian dict (already filtered to non-sky pixels)
        sky_mask_strided: (n_rows, n_cols) bool sky mask at stride resolution
        depth_map_strided: (n_rows, n_cols) depth at stride resolution
        erp_image_strided: (n_rows, n_cols, 3) uint8 image at stride resolution
        sky_mode: BACKGROUND_SPHERE or LOW_OPACITY
        depth_max: Scene depth max (sky sphere radius default)
        sky_radius: Fixed radius for BACKGROUND_SPHERE mode
        sky_opacity: Opacity applied to sky Gaussians

    Returns:
        gaussians dict with sky Gaussians appended (or unchanged if mode=SKIP)
    """
    if sky_mode == SkyMode.SKIP:
        return gaussians

    if not sky_mask_strided.any():
        return gaussians

    # Sky pixel positions at stride resolution
    sky_rows, sky_cols = np.where(sky_mask_strided)
    n_sky = len(sky_rows)

    if n_sky == 0:
        return gaussians

    H_s, W_s = sky_mask_strided.shape

    # Spherical coordinates of sky pixels
    theta = sky_rows / H_s * np.pi        # colatitude 0…π
    phi   = sky_cols / W_s * 2 * np.pi    # longitude 0…2π

    if sky_mode == SkyMode.BACKGROUND_SPHERE:
        r = np.full(n_sky, sky_radius)
    else:  # LOW_OPACITY — use near-depth_max distance
        r = np.full(n_sky, depth_max * 1.5)

    # 3D positions on sphere (Y-up, negative Z forward)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = -r * np.sin(theta) * np.sin(phi)
    positions = np.stack([x, y, z], axis=-1).astype(np.float32)   # (N_sky, 3)

    # Colors from image
    colors = erp_image_strided[sky_rows, sky_cols].astype(np.float32) / 255.0

    # Large flat billboards for sky
    scale_base = sky_radius * 0.04 * np.ones(n_sky, dtype=np.float32)
    scales = np.stack([scale_base, scale_base, scale_base * 0.01], axis=-1)

    # Identity quaternion (isotropic orientation) in XYZW order
    quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_sky, 1))

    opacities = np.full((n_sky, 1), sky_opacity, dtype=np.float32)
    sh1 = np.zeros((n_sky, 9), dtype=np.float32)

    import torch
    if device is None:
        device = gaussians['means'].device

    sky_dict = {
        'means':     torch.from_numpy(positions).to(device),
        'scales':    torch.from_numpy(scales).to(device),
        'quats':     torch.from_numpy(quats).to(device),
        'colors':    torch.from_numpy(colors).to(device),
        'opacities': torch.from_numpy(opacities).to(device),
        'sh1':       torch.from_numpy(sh1).to(device),
    }

    # Concatenate with existing gaussians
    combined = {}
    common_keys = set(gaussians.keys()) & set(sky_dict.keys())
    for key in common_keys:
        combined[key] = torch.cat([gaussians[key], sky_dict[key]], dim=0)
    # Keep any extra keys from base gaussians
    for key in gaussians.keys():
        if key not in combined:
            combined[key] = gaussians[key]

    return combined

def prune_grazing_angle(
    gaussians: dict,
    depth_map: np.ndarray,
    stride: int = 2,
    max_angle_deg: float = 80.0,
) -> dict:
    """
    Remove Gaussians at extreme grazing angles where the depth surface
    is nearly edge-on to the camera ray. These create the "striation"
    artifacts behind rocks and objects where depth wraps around edges.

    Works by computing the local depth gradient — steep depth changes
    between adjacent pixels indicate a surface nearly parallel to the
    viewing ray. These splats fan out into elongated strips.

    Args:
        gaussians: Dict of Gaussian tensors (means, scales, etc.)
        depth_map: (H, W) original depth map (numpy float32)
        stride: Pixel stride used during conversion
        max_angle_deg: Maximum grazing angle in degrees (default 80).
            Lower = more aggressive pruning of edge splats.
            90 = no filtering, 60 = aggressive.

    Returns:
        Filtered gaussians dict
    """
    import torch
    if max_angle_deg >= 90.0 or gaussians['means'].shape[0] == 0:
        return gaussians

    H, W = depth_map.shape

    # Compute depth gradient magnitude (Sobel-like)
    # Gradient in theta (horizontal) and phi (vertical) directions
    pad_depth = np.pad(depth_map, 1, mode='wrap')  # wrap horizontally for 360°
    grad_y = (pad_depth[2:, 1:-1] - pad_depth[:-2, 1:-1]) / 2.0
    grad_x = (pad_depth[1:-1, 2:] - pad_depth[1:-1, :-2]) / 2.0
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Relative gradient: gradient / depth (dimensionless)
    safe_depth = np.maximum(depth_map, 0.01)
    relative_grad = grad_mag / safe_depth

    # Convert max_angle to relative gradient threshold
    # tan(angle) ≈ depth_gradient / (depth * angular_spacing)
    # For a surface at angle θ from normal: relative_grad ≈ tan(θ) * angular_spacing
    angular_spacing = np.pi / H * stride
    max_relative_grad = np.tan(np.radians(max_angle_deg)) * angular_spacing

    # Sample gradient at Gaussian positions
    means = gaussians['means'].detach().cpu().numpy()
    N = means.shape[0]

    # Back-project positions to pixel coordinates
    # positions = depth * rhat, rhat = [sin(phi)cos(theta), cos(phi), -sin(phi)sin(theta)]
    r = np.linalg.norm(means, axis=1)
    y = means[:, 1]
    phi = np.arccos(np.clip(y / np.maximum(r, 1e-8), -1, 1))  # [0, pi]
    theta = np.arctan2(-means[:, 2], means[:, 0])  # [-pi, pi]
    theta = theta % (2 * np.pi)  # [0, 2pi]

    # Map to pixel coordinates
    px_row = np.clip((phi / np.pi * H).astype(int), 0, H - 1)
    px_col = np.clip((theta / (2 * np.pi) * W).astype(int), 0, W - 1)

    # Sample relative gradient at each Gaussian's pixel
    sampled_grad = relative_grad[px_row, px_col]

    keep_mask_np = sampled_grad < max_relative_grad
    keep_mask = torch.from_numpy(keep_mask_np).to(gaussians['means'].device)

    pruned = {}
    for key, tensor in gaussians.items():
        pruned[key] = tensor[keep_mask]

    removed = N - int(keep_mask_np.sum())
    if removed > 0:
        print(f"[Grazing Angle] Removed {removed:,} edge splats (max_angle={max_angle_deg}°)")

    return pruned


def prune_sparse_regions(
    gaussians: dict,
    min_neighbors: int = 3,
    radius_multiplier: float = 3.0,
    k: int = 8,
) -> dict:
    """
    Remove Gaussians in low-density regions where local spacing far
    exceeds the expected density. Unlike SOR (which uses global statistics),
    this compares each splat's neighbor distance against its own scale,
    removing splats that are isolated relative to their expected size.

    Targets the "finger" artifacts where splats spread apart behind objects
    with increasing spacing, while preserving legitimately sparse regions
    like distant backgrounds.

    Args:
        gaussians: Dict of Gaussian tensors
        min_neighbors: Minimum neighbors within the adaptive radius.
            Splats with fewer neighbors are removed. (default 3)
        radius_multiplier: Search radius = splat_scale * this multiplier.
            Higher = more lenient. (default 3.0)
        k: Number of nearest neighbors to check. (default 8)

    Returns:
        Filtered gaussians dict
    """
    import torch

    N = gaussians['means'].shape[0]
    if N < k + 1:
        return gaussians

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        import warnings
        warnings.warn("scipy not installed. Skipping sparse region pruning.")
        return gaussians

    means_np = gaussians['means'].detach().cpu().numpy()
    scales_np = gaussians['scales'].detach().cpu().numpy()

    # Use the max scale dimension as the expected splat radius
    splat_radius = np.max(scales_np, axis=1)  # [N]

    tree = cKDTree(means_np)

    # For each splat, count how many neighbors are within radius_multiplier * scale
    search_radius = splat_radius * radius_multiplier
    # Clamp to reasonable range to avoid degenerate queries
    search_radius = np.clip(search_radius, 0.001, 100.0)

    # Query k nearest neighbors
    distances, _ = tree.query(means_np, k=k + 1, workers=-1)
    neighbor_dists = distances[:, 1:]  # exclude self

    # Count neighbors within the adaptive radius
    within_radius = neighbor_dists < search_radius[:, np.newaxis]
    neighbor_count = np.sum(within_radius, axis=1)

    keep_mask_np = neighbor_count >= min_neighbors
    keep_mask = torch.from_numpy(keep_mask_np).to(gaussians['means'].device)

    pruned = {}
    for key, tensor in gaussians.items():
        pruned[key] = tensor[keep_mask]

    removed = N - int(keep_mask_np.sum())
    if removed > 0:
        print(f"[Sparse Regions] Removed {removed:,} isolated splats (min_neighbors={min_neighbors})")

    return pruned


def prune_outliers(
    gaussians: dict,
    strength: float = 0.5,
    k: int = 16,
) -> dict:
    """
    Remove isolated Gaussians (floaters) using Statistical Outlier Removal (SOR).
    
    Args:
        gaussians: Dict of Gaussian tensors (means, scales, opacities, colors, quats, sh1)
        strength: Rejection strength from 0.0 to 1.0 (higher = more aggressive pruning)
        k: Number of nearest neighbors to consider
        
    Returns:
        Pruned gaussians dict
    """
    if strength <= 0.0 or gaussians['means'].shape[0] < k + 1:
        return gaussians
        
    import numpy as np
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        import warnings
        warnings.warn("scipy not installed. Skipping outlier pruning.")
        return gaussians

    means_np = gaussians['means'].detach().cpu().numpy()
    tree = cKDTree(means_np)
    
    # Query distances to k nearest neighbors (k+1 because the point itself is included at dist=0)
    # n_jobs=-1 uses all CPU cores
    distances, _ = tree.query(means_np, k=k + 1, workers=-1)
    
    # Average distance to neighbors (excluding the point itself)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    
    # Compute global mean and std
    global_mean = np.mean(avg_distances)
    global_std = np.std(avg_distances)
    
    # Mapping strength (0.0 - 1.0) to std_ratio (3.0 to 0.5)
    # std_ratio defines how many standard deviations away from the mean is acceptable.
    # lower std_ratio = more aggressive pruning.
    # strength=0.0 -> std_ratio=3.0 (mild)
    # strength=1.0 -> std_ratio=0.5 (aggressive)
    std_ratio = 3.0 - (strength * 2.5) 
    
    threshold = global_mean + (global_std * std_ratio)
    
    # Keep points whose average distance is below the threshold
    keep_mask_np = avg_distances < threshold
    
    import torch
    keep_mask = torch.from_numpy(keep_mask_np).to(gaussians['means'].device)
    
    pruned = {}
    for key, tensor in gaussians.items():
        pruned[key] = tensor[keep_mask]
        
    removed_count = len(means_np) - int(keep_mask_np.sum())
    print(f"[Outlier Pruning] Removed {removed_count:,} floaters (strength={strength:.2f})")
    
    return pruned
