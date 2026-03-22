"""Three-way region classification: warp vs splat vs panorama."""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation


class RegionType(IntEnum):
    """Region classification types."""
    TRUSTED = 0    # Warp + splat agree — geometry is correct
    TYPE_A = 1     # Both valid but disagree — degraded geometry
    TYPE_B = 2     # Warp invalid, splat weak — uncertain
    TYPE_C = 3     # Confirmed disocclusion — needs repair


def classify_frame(
    warp_rgb: np.ndarray,
    warp_valid: np.ndarray,
    splat_rgb: np.ndarray,
    splat_alpha: np.ndarray,
    rgb_error_threshold: float = 0.08,
    alpha_gap_threshold: float = 0.3,
    type_c_alpha_max: float = 0.1,
    transition_band: int = 5,
    pano_rgb: Optional[np.ndarray] = None,
    pano_valid: Optional[np.ndarray] = None,
    splat_depth: Optional[np.ndarray] = None,
    warp_depth: Optional[np.ndarray] = None,
    depth_disagreement_threshold: float = 0.15,
) -> np.ndarray:
    """
    Classify each pixel into one of four region types.

    Three-way comparison (spec Section 6.1):
    - Warp valid + splat valid + agree → TRUSTED
    - Warp valid + splat valid + disagree → TYPE_A (degraded geometry)
    - Warp valid + splat low alpha → TYPE_C (confirmed disocclusion)
    - Neither valid → TYPE_C (truly unobserved)
    - Warp invalid + splat weak → TYPE_B

    When optional panoramic extraction is provided, pixels where all three
    sources (warp + splat + pano) agree get stronger TRUSTED classification.
    When depth maps are provided, relative depth disagreement contributes to
    TYPE_A detection.

    Args:
        warp_rgb: [H, W, 3] forward-warped RGB
        warp_valid: [H, W] bool — True where warp has valid content
        splat_rgb: [H, W, 3] gsplat rendered RGB
        splat_alpha: [H, W] gsplat rendered alpha
        rgb_error_threshold: Max L1 error for agreement
        alpha_gap_threshold: Alpha below this = "weak" splat
        type_c_alpha_max: Alpha below this = confirmed gap
        transition_band: Dilation pixels for transition zones
        pano_rgb: [H, W, 3] optional panoramic extraction RGB
        pano_valid: [H, W] optional bool mask for pano extraction
        splat_depth: [H, W] optional rendered depth
        warp_depth: [H, W] optional forward-warped depth
        depth_disagreement_threshold: Relative depth error for TYPE_A

    Returns:
        [H, W] int array with RegionType values
    """
    H, W = warp_valid.shape
    result = np.full((H, W), RegionType.TYPE_C, dtype=np.int32)

    splat_valid = splat_alpha > alpha_gap_threshold
    splat_weak = (splat_alpha > type_c_alpha_max) & (splat_alpha <= alpha_gap_threshold)
    splat_gap = splat_alpha <= type_c_alpha_max

    # RGB agreement check (L1 distance per pixel)
    rgb_error = np.mean(np.abs(warp_rgb - splat_rgb), axis=-1)

    # Classification logic
    both_valid = warp_valid & splat_valid
    agree = rgb_error < rgb_error_threshold

    # TRUSTED: both valid and agree
    result[both_valid & agree] = RegionType.TRUSTED

    # TYPE_A: both valid but disagree
    result[both_valid & ~agree] = RegionType.TYPE_A

    # TYPE_C: warp valid but splat is a gap (confirmed disocclusion)
    result[warp_valid & splat_gap] = RegionType.TYPE_C

    # TYPE_B: warp invalid, splat weak
    result[~warp_valid & splat_weak] = RegionType.TYPE_B

    # TYPE_C: neither valid (truly unobserved)
    result[~warp_valid & splat_gap] = RegionType.TYPE_C

    # --- Three-way panoramic agreement strengthening ---
    if pano_rgb is not None and pano_valid is not None:
        pano_error = np.mean(np.abs(pano_rgb - splat_rgb), axis=-1)
        pano_warp_error = np.mean(np.abs(pano_rgb - warp_rgb), axis=-1)
        three_way_agree = (
            pano_valid & warp_valid & splat_valid
            & (pano_error < rgb_error_threshold)
            & (pano_warp_error < rgb_error_threshold)
        )
        # Strengthen: pixels with three-way agreement stay TRUSTED
        # even if they'd otherwise be borderline TYPE_A
        result[three_way_agree] = RegionType.TRUSTED

    # --- Depth-based TYPE_A detection ---
    if splat_depth is not None and warp_depth is not None:
        depth_valid = warp_valid & splat_valid & (warp_depth > 0.01)
        relative_depth_error = np.zeros((H, W), dtype=np.float32)
        relative_depth_error[depth_valid] = (
            np.abs(splat_depth[depth_valid] - warp_depth[depth_valid])
            / warp_depth[depth_valid]
        )
        depth_disagree = depth_valid & (relative_depth_error > depth_disagreement_threshold)
        # Pixels that agree on RGB but disagree on depth → TYPE_A
        result[depth_disagree & (result == RegionType.TRUSTED)] = RegionType.TYPE_A

    # Add transition band around TYPE_C regions
    if transition_band > 0:
        type_c_mask = result == RegionType.TYPE_C
        dilated = binary_dilation(type_c_mask, iterations=transition_band)
        transition = dilated & ~type_c_mask & (result == RegionType.TRUSTED)
        result[transition] = RegionType.TYPE_B

    return result
