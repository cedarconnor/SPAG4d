# spag4d/depth_blend.py
"""
Depth map fusion and blending utilities for SPAG-4D.

Provides three blending modes ordered by quality/cost:
  1. FEATHERED  — simple edge-distance weighted average (fast, Phase 1 default)
  2. LAPLACIAN  — Laplacian pyramid frequency-band fusion (recommended)
  3. POISSON    — gradient-domain Poisson blending (best quality, slowest)

Primary use case: fuse a globally-consistent low-frequency DAP depth map with
a high-frequency detail depth map (from Depth Pro, DA3, or similar perspective
models) to get the best of both.

All functions work on numpy float32 arrays in [0, ∞) metre space.
Requires: numpy, scipy, opencv-python.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import List, Optional, Tuple


class BlendMode(Enum):
    FEATHERED = "feathered"  # Simple, fast
    LAPLACIAN = "laplacian"  # Frequency-domain fusion (recommended)
    POISSON   = "poisson"    # Gradient-domain, slowest, best seam removal


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian / Laplacian Pyramid helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_gaussian_pyramid(image: np.ndarray, n_levels: int) -> List[np.ndarray]:
    """Standard Gaussian pyramid via iterative blur + downsample."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for pyramid blending. pip install opencv-python")

    pyramid = [image.astype(np.float32)]
    current = image.astype(np.float32)
    for _ in range(n_levels - 1):
        blurred    = cv2.GaussianBlur(current, (5, 5), 1.0)
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid


def _build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """L[i] = G[i] - upsample(G[i+1])"""
    import cv2
    laplacian = []
    for i in range(len(gaussian_pyramid) - 1):
        g_fine   = gaussian_pyramid[i]
        g_coarse = gaussian_pyramid[i + 1]
        upsampled = cv2.resize(g_coarse, (g_fine.shape[1], g_fine.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        laplacian.append(g_fine - upsampled)
    laplacian.append(gaussian_pyramid[-1])   # Coarsest level is a Gaussian (residual)
    return laplacian


def _reconstruct_from_laplacian(laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    import cv2
    current = laplacian_pyramid[-1].copy()
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        finer = laplacian_pyramid[i]
        upsampled = cv2.resize(current, (finer.shape[1], finer.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        current = upsampled + finer
    return current


# ──────────────────────────────────────────────────────────────────────────────
# Public Fusion Functions
# ──────────────────────────────────────────────────────────────────────────────

def laplacian_depth_fusion(
    dap_depth: np.ndarray,
    dp_depth: np.ndarray,
    n_levels: int = 5,
    low_freq_cutoff: int = 2,
) -> np.ndarray:
    """
    Fuse two depth maps using Laplacian pyramid blending.

    Low pyramid levels (coarse structure) come from `dap_depth` which has
    equirectangular-aware global consistency. High levels (fine edges/detail)
    come from `dp_depth` (Depth Pro, DA3, or similar perspective-model depth).

    Args:
        dap_depth:       (H, W) float32 — globally-consistent depth (e.g. DAP)
        dp_depth:        (H, W) float32 — high-detail depth (e.g. Depth Pro)
        n_levels:        Pyramid depth
        low_freq_cutoff: Levels 0..cutoff use dap_depth; cutoff+1..n use dp_depth

    Returns:
        fused: (H, W) float32 fused depth
    """
    dap_lap = _build_laplacian_pyramid(_build_gaussian_pyramid(dap_depth, n_levels))
    dp_lap  = _build_laplacian_pyramid(_build_gaussian_pyramid(dp_depth,  n_levels))

    fused_lap = []
    for level in range(n_levels):
        # Level 0 is the finest detail, Level n-1 is the coarsest residual.
        if level >= n_levels - low_freq_cutoff:
            fused_lap.append(dap_lap[level])   # coarse: from DAP
        else:
            fused_lap.append(dp_lap[level])    # fine:   from Depth Pro

    fused = _reconstruct_from_laplacian(fused_lap)
    return fused.clip(0.0, None).astype(np.float32)


def masked_laplacian_fusion(
    dap_depth: np.ndarray,
    dp_depth: np.ndarray,
    dp_confidence: np.ndarray,
    n_levels: int = 5,
) -> np.ndarray:
    """
    Confidence-weighted Laplacian pyramid fusion.

    At each pyramid level, the blend weight between the two depth maps is
    controlled by a downsampled version of `dp_confidence`. Where the detail
    model is confident (sharp surfaces) it contributes more; where it is
    uncertain (sky, reflections) the DAP depth dominates.

    Args:
        dap_depth:      (H, W) float32 — globally-consistent depth
        dp_depth:       (H, W) float32 — high-detail depth
        dp_confidence:  (H, W) float32 in [0, 1] — Depth Pro / DA3 confidence
        n_levels:       Pyramid depth

    Returns:
        fused: (H, W) float32
    """
    import cv2

    dap_lap  = _build_laplacian_pyramid(_build_gaussian_pyramid(dap_depth,      n_levels))
    dp_lap   = _build_laplacian_pyramid(_build_gaussian_pyramid(dp_depth,       n_levels))
    conf_pyr = _build_gaussian_pyramid(dp_confidence.astype(np.float32), n_levels)

    fused_lap = []
    for level in range(n_levels):
        conf = conf_pyr[min(level, len(conf_pyr) - 1)]
        dap_l = dap_lap[level]
        dp_l  = dp_lap[level]

        # Resize confidence to match this level
        if conf.shape != dap_l.shape:
            conf = cv2.resize(conf, (dap_l.shape[1], dap_l.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        # Reverse logic: lower levels (fing detail) should have DP weighted higher
        # so we multiply confidence directly against DP detail
        fused_l = conf * dp_l + (1.0 - conf) * dap_l
        fused_lap.append(fused_l)

    fused = _reconstruct_from_laplacian(fused_lap)
    return fused.clip(0.0, None).astype(np.float32)


def feathered_blend(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    weight_a: np.ndarray,
) -> np.ndarray:
    """
    Simple weighted average (alpha blend) of two depth maps.

    Args:
        depth_a:  (H, W) float32
        depth_b:  (H, W) float32
        weight_a: (H, W) float32 in [0, 1] — weight for depth_a

    Returns:
        blended: (H, W) float32
    """
    w = np.clip(weight_a.astype(np.float32), 0.0, 1.0)
    return (w * depth_a + (1.0 - w) * depth_b).astype(np.float32)


def poisson_blend_faces(
    face_depths: dict,
    face_uv_maps: dict,
    target_shape: Tuple[int, int],
    dap_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Gradient-domain (Poisson) blending of multiple depth face maps onto ERP.

    Composites depth GRADIENTS from each face (weighted by edge distance),
    then solves the Poisson equation to find the depth field that best matches
    the composed gradients. This gives mathematically optimal seam removal.

    Falls back to simple feathered composite if sparse-solver is unavailable.

    Args:
        face_depths:  dict[str, (H_f, W_f)] aligned depth per face
        face_uv_maps: dict[str, (H_f, W_f, 2)] ERP pixel coords per face
        target_shape: (H, W) target ERP resolution
        dap_depth:    (H, W) DAP depth used as Dirichlet boundary / fallback

    Returns:
        solved_depth: (H, W) float32
    """
    H, W = target_shape

    gradient_x  = np.zeros((H, W), dtype=np.float64)
    gradient_y  = np.zeros((H, W), dtype=np.float64)
    weight_map  = np.zeros((H, W), dtype=np.float64)
    composite   = np.zeros((H, W), dtype=np.float64)

    def _edge_weight(face_size: int) -> np.ndarray:
        """Cosine-like weight: 1 at centre, 0 at border."""
        r = np.arange(face_size)
        u = np.minimum(r, r[::-1]).astype(np.float64)
        half = face_size / 2.0
        w = np.clip(u / half, 0.0, 1.0)  # linear 0→1→0
        uu, vv = np.meshgrid(w, w)
        return np.minimum(uu, vv)

    for face_dir, depth in face_depths.items():
        uv  = face_uv_maps[face_dir]           # (H_f, W_f, 2)
        H_f = depth.shape[0]
        w   = _edge_weight(H_f)                # (H_f, W_f)

        gx = np.gradient(depth.astype(np.float64), axis=1)
        gy = np.gradient(depth.astype(np.float64), axis=0)

        erp_x = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
        erp_y = np.round(uv[..., 1]).astype(int).clip(0, H - 1)

        np.add.at(gradient_x,  (erp_y, erp_x), gx * w)
        np.add.at(gradient_y,  (erp_y, erp_x), gy * w)
        np.add.at(weight_map,  (erp_y, erp_x), w)
        np.add.at(composite,   (erp_y, erp_x), depth * w)

    # Normalise accumulated gradients
    safe_w = np.where(weight_map > 1e-8, weight_map, 1.0)
    gradient_x /= safe_w
    gradient_y /= safe_w
    composite  /= safe_w

    # Attempt Poisson solve via sparse linear system
    try:
        from scipy.sparse     import lil_matrix
        from scipy.sparse.linalg import spsolve

        N = H * W
        A = lil_matrix((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        div = (np.gradient(gradient_x, axis=1) + np.gradient(gradient_y, axis=0))

        for row in range(H):
            for col in range(W):
                idx = row * W + col
                # Boundary condition: use dap_depth (or composite) if available
                if row == 0 or row == H - 1 or col == 0 or col == W - 1:
                    A[idx, idx] = 1.0
                    if dap_depth is not None:
                        b[idx] = float(dap_depth[row, col])
                    else:
                        b[idx] = float(composite[row, col])
                    continue

                A[idx, idx]               =  4.0
                A[idx, idx - 1]           = -1.0
                A[idx, idx + 1]           = -1.0
                A[idx, idx - W]           = -1.0
                A[idx, idx + W]           = -1.0
                b[idx] = div[row, col]

        solved = spsolve(A.tocsr(), b)
        return solved.reshape(H, W).clip(0.0, None).astype(np.float32)

    except (ImportError, Exception):
        # Fallback: simple composite
        if dap_depth is not None:
            gaps = weight_map < 0.01
            composite_f = composite.astype(np.float32)
            composite_f[gaps] = dap_depth[gaps].astype(np.float32)
            return composite_f
        return composite.clip(0.0, None).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Class
# ──────────────────────────────────────────────────────────────────────────────

class DepthBlender:
    """
    High-level wrapper for all depth fusion methods.

    Usage::

        blender = DepthBlender(mode=BlendMode.LAPLACIAN)
        fused   = blender.fuse(dap_depth, detail_depth, confidence=conf_map)
    """

    def __init__(
        self,
        mode: BlendMode = BlendMode.LAPLACIAN,
        n_levels: int = 5,
        low_freq_cutoff: int = 2,
    ):
        self.mode            = mode
        self.n_levels        = n_levels
        self.low_freq_cutoff = low_freq_cutoff

    def fuse(
        self,
        dap_depth: np.ndarray,
        detail_depth: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse globally-consistent DAP depth with high-detail per-face depth.

        Args:
            dap_depth:    (H, W) float32 — DAP / PanDA global depth
            detail_depth: (H, W) float32 — high-detail depth (Depth Pro, etc.)
            confidence:   (H, W) float32 [0,1] — optional per-pixel confidence

        Returns:
            fused: (H, W) float32
        """
        if self.mode == BlendMode.LAPLACIAN:
            if confidence is not None:
                return masked_laplacian_fusion(
                    dap_depth, detail_depth, confidence, n_levels=self.n_levels
                )
            return laplacian_depth_fusion(
                dap_depth, detail_depth,
                n_levels=self.n_levels,
                low_freq_cutoff=self.low_freq_cutoff,
            )

        elif self.mode == BlendMode.FEATHERED:
            if confidence is not None:
                return feathered_blend(detail_depth, dap_depth, confidence)
            # Equal 50/50 blend if no confidence available
            return feathered_blend(detail_depth, dap_depth,
                                   np.full_like(dap_depth, 0.5))

        elif self.mode == BlendMode.POISSON:
            raise ValueError(
                "Use poisson_blend_faces() directly for cubemap/tangent patch blending. "
                "DepthBlender.fuse() does not support POISSON mode (requires face UV maps)."
            )

        raise ValueError(f"Unknown blend mode: {self.mode}")

    def blend_cubemap_faces(
        self,
        face_depths: dict,
        face_uv_maps: dict,
        target_shape: Tuple[int, int],
        dap_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite multiple cubemap face depths back to ERP space."""
        if self.mode == BlendMode.POISSON:
            return poisson_blend_faces(face_depths, face_uv_maps, target_shape, dap_depth)

        # Feathered fallback for LAPLACIAN and FEATHERED modes
        H, W = target_shape
        depth_sum  = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        for face_dir, depth in face_depths.items():
            uv   = face_uv_maps[face_dir]
            H_f  = depth.shape[0]
            half = H_f / 2.0
            r    = np.arange(H_f)
            u    = np.minimum(r, r[::-1]).astype(np.float32)
            uu, vv = np.meshgrid(u / half, u / half)
            w = np.minimum(uu, vv).clip(0.0, 1.0)

            erp_x = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
            erp_y = np.round(uv[..., 1]).astype(int).clip(0, H - 1)

            np.add.at(depth_sum,  (erp_y, erp_x), depth.astype(np.float32) * w)
            np.add.at(weight_sum, (erp_y, erp_x), w)

        result = depth_sum / np.where(weight_sum > 1e-8, weight_sum, 1.0)

        if dap_depth is not None:
            gaps = weight_sum < 0.01
            result[gaps] = dap_depth.astype(np.float32)[gaps]

        return result
