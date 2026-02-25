# spag4d/adaptive_stride.py
"""
Depth-adaptive stride computation and variable-stride Gaussian position sampling.

Instead of uniform pixel-grid sampling, this module:
  1. Computes per-pixel stride proportional to depth (closer = denser)
  2. Optionally reduces stride near depth discontinuities (edge-aware)
  3. Samples (row, col) positions using this variable stride map
  4. Provides a binary-search helper to hit a target Gaussian count

All inputs/outputs use numpy for CPU-friendly, pre-GPU processing.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Stride Map Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_adaptive_stride_map(
    depth_map: np.ndarray,
    base_stride: int = 2,
    min_stride: int = 1,
    max_stride: int = 8,
    depth_reference: float = 5.0,
) -> np.ndarray:
    """
    Compute per-pixel stride proportional to depth.

    Closer objects (depth < depth_reference) get stride ≤ base_stride.
    Farther objects get larger stride up to max_stride.

    stride(d) = round(base_stride * d / depth_reference), clamped to [min, max].

    Args:
        depth_map:       (H, W) float32 depth in metres
        base_stride:     Stride at depth_reference distance
        min_stride:      Minimum (most dense) stride
        max_stride:      Maximum (most sparse) stride
        depth_reference: Depth at which stride == base_stride

    Returns:
        stride_map: (H, W) int32 — per-pixel stride value
    """
    stride_float = base_stride * (depth_map / max(depth_reference, 1e-3))
    stride_int   = np.round(stride_float).astype(np.int32)
    stride_int   = np.clip(stride_int, min_stride, max_stride)
    return stride_int


def refine_stride_at_edges(
    stride_map: np.ndarray,
    depth_map: np.ndarray,
    edge_reduction_factor: float = 0.5,
    canny_low: int = 30,
    canny_high: int = 100,
    dilation_size: int = 7,
    dilation_iters: int = 2,
) -> np.ndarray:
    """
    Reduce stride in regions near depth discontinuities (object edges).

    Detects edges via Canny applied to the normalised depth map, dilates
    the edge mask to form a border zone, then halves the stride there.

    Requires opencv-python.

    Args:
        stride_map:           (H, W) int32 stride from compute_adaptive_stride_map
        depth_map:            (H, W) float32 depth for edge detection
        edge_reduction_factor: Stride multiplier in edge zones (0.5 = half stride)
        canny_low/high:       Canny hysteresis thresholds
        dilation_size:        Dilation kernel size (pixels)
        dilation_iters:       Number of dilation iterations

    Returns:
        refined_stride_map: (H, W) int32
    """
    try:
        import cv2
    except ImportError:
        return stride_map  # graceful fallback if cv2 unavailable

    # Normalise depth to 0-255 for Canny
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max > d_min:
        depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        return stride_map  # flat depth — no edges

    edges = cv2.Canny(depth_norm, canny_low, canny_high)

    kernel   = np.ones((dilation_size, dilation_size), dtype=np.uint8)
    edge_zone = cv2.dilate(edges, kernel, iterations=dilation_iters) > 0

    refined = stride_map.copy()
    # Reduce stride in edge zone, keep minimum at 1
    refined[edge_zone] = np.maximum(
        np.round(refined[edge_zone] * edge_reduction_factor).astype(np.int32),
        1,
    )
    return refined


# ──────────────────────────────────────────────────────────────────────────────
# Variable-Stride Position Sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_with_adaptive_stride(
    stride_map: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Collect (row, col) Gaussian positions using a per-pixel stride map.

    Uses a greedy row-by-row walk:
      - Row advance = median stride of the current row
      - Column advance = local stride at current pixel

    `sky_mask` and `keep_mask` are at FULL resolution (same shape as stride_map).
    This function skips pixels where sky_mask=True or keep_mask=False.

    Args:
        stride_map: (H, W) int32 per-pixel stride
        sky_mask:   (H, W) bool — True = sky, skip
        keep_mask:  (H, W) bool — True = valid, keep

    Returns:
        positions: list of (row, col) tuples
    """
    H, W = stride_map.shape

    if sky_mask is None:
        sky_mask = np.zeros((H, W), dtype=bool)
    if keep_mask is None:
        keep_mask = np.ones((H, W), dtype=bool)

    positions: List[Tuple[int, int]] = []

    row = 0
    while row < H:
        # Row-level stride: use median to handle outliers at depth discontinuities
        row_stride = int(np.median(stride_map[row, :]))
        row_stride = max(row_stride, 1)

        col = 0
        while col < W:
            local_stride = int(stride_map[row, col])
            local_stride = max(local_stride, 1)

            if not sky_mask[row, col] and keep_mask[row, col]:
                positions.append((row, col))

            col += local_stride

        row += row_stride

    return positions


def sample_uniform(
    H: int,
    W: int,
    stride: int,
    sky_mask: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Uniform-stride sampling with optional sky/validity masking.

    Drop-in replacement for `sample_with_adaptive_stride` when
    adaptive_stride is disabled. Produces the same (row, col) list format.

    Args:
        H, W:      Full-resolution image dimensions
        stride:    Uniform pixel skip
        sky_mask:  (H, W) bool at full resolution — True = skip
        keep_mask: (H, W) bool at full resolution — True = generate Gaussian

    Returns:
        positions: list of (row, col) tuples
    """
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)

    positions: List[Tuple[int, int]] = []
    for r in rows:
        for c in cols:
            if sky_mask is not None and sky_mask[r, c]:
                continue
            if keep_mask is not None and not keep_mask[r, c]:
                continue
            positions.append((int(r), int(c)))

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# Budget-Aware Base Stride Estimation
# ──────────────────────────────────────────────────────────────────────────────

def compute_stride_for_budget(
    depth_map: np.ndarray,
    target_count: int,
    sky_mask: Optional[np.ndarray] = None,
    min_stride: int = 1,
    max_stride: int = 8,
    pole_thinning_factor: float = 0.70,
) -> int:
    """
    Estimate the base_stride that produces approximately target_count Gaussians.

    Uses the approximation:  count ≈ valid_pixels * pole_thinning_factor / stride²

    Then refines with a quick integer sweep if the approximation overshoots.

    Args:
        depth_map:             (H, W) depth map
        target_count:          Desired Gaussian count
        sky_mask:              (H, W) bool — sky pixels excluded from budget
        min_stride / max_stride: Allowed stride range
        pole_thinning_factor:  Approximate fraction of pixels kept after pole thinning

    Returns:
        base_stride: int
    """
    H, W = depth_map.shape
    total_pixels = H * W

    if sky_mask is not None:
        if sky_mask.shape != (H, W):
            import cv2
            sky_mask = cv2.resize(
                sky_mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        valid_pixels = int((~sky_mask).sum())
    else:
        valid_pixels = total_pixels

    # Approximate: grid cells = valid_pixels / stride²; thinning removes ~30%
    approx_stride_f = np.sqrt(valid_pixels * pole_thinning_factor / max(target_count, 1))
    approx_stride   = int(np.clip(round(approx_stride_f), min_stride, max_stride))

    # Quick sweep to refine (5 candidates around the estimate)
    candidates = sorted(set(
        np.clip(np.arange(approx_stride - 2, approx_stride + 3), min_stride, max_stride).tolist()
    ))

    best_stride = approx_stride
    best_diff   = float("inf")

    for s in candidates:
        estimated = valid_pixels * pole_thinning_factor / (s * s)
        diff = abs(estimated - target_count)
        if diff < best_diff:
            best_diff   = diff
            best_stride = s

    return best_stride
