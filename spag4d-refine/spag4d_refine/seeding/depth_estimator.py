"""Monocular depth estimation + forward-warp alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignedDepth:
    """Depth estimation result aligned to the forward-warped depth."""
    depth: np.ndarray          # [H, W] aligned Z-depth
    confidence: np.ndarray     # [H, W] float [0, 1]
    scale: float               # Affine alignment scale
    offset: float              # Affine alignment offset


def estimate_depth(
    rgb: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Estimate monocular depth using DepthAnything V2 via transformers.

    Args:
        rgb: [H, W, 3] float32 [0, 1]
        device: Torch device

    Returns:
        [H, W] float32 relative depth (not metric)
    """
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    from PIL import Image

    model_name = "depth-anything/Depth-Anything-V2-Small-hf"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    # Convert to PIL
    pil_image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(rgb.shape[0], rgb.shape[1]),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth.astype(np.float32)


def estimate_and_align_depth(
    synthesized_rgb: np.ndarray,
    warp_depth: np.ndarray,
    warp_valid: np.ndarray,
    gap_mask: np.ndarray,
    device: str = "cuda",
    confidence_decay_pixels: int = 50,
) -> AlignedDepth:
    """
    Estimate depth on synthesized RGB and align to forward-warped depth.

    Affine alignment: warp_depth ≈ scale * estimated + offset
    Solved by least squares at overlap pixels (where warp is valid).

    Args:
        synthesized_rgb: [H, W, 3] repaired RGB
        warp_depth: [H, W] forward-warped Z-depth
        warp_valid: [H, W] bool — where warp depth is reliable
        gap_mask: [H, W] bool — True at gap regions
        device: Torch device
        confidence_decay_pixels: Confidence decay distance from known depth

    Returns:
        AlignedDepth with metric-aligned depth and confidence map
    """
    # Estimate relative depth
    raw_depth = estimate_depth(synthesized_rgb, device=device)

    # Affine alignment at overlap (non-gap, warp-valid) pixels
    overlap = warp_valid & ~gap_mask
    if overlap.sum() < 10:
        logger.warning("Too few overlap pixels for depth alignment, using raw depth")
        return AlignedDepth(
            depth=raw_depth,
            confidence=np.ones_like(raw_depth) * 0.5,
            scale=1.0,
            offset=0.0,
        )

    # Least squares: warp_depth = scale * raw_depth + offset
    A = np.stack([raw_depth[overlap], np.ones(overlap.sum())], axis=-1)
    b = warp_depth[overlap]

    # Robust: use only finite values
    finite = np.isfinite(b) & np.isfinite(A[:, 0])
    A = A[finite]
    b = b[finite]

    if len(b) < 10:
        return AlignedDepth(
            depth=raw_depth,
            confidence=np.ones_like(raw_depth) * 0.5,
            scale=1.0,
            offset=0.0,
        )

    result = np.linalg.lstsq(A, b, rcond=None)
    scale, offset = result[0]

    # Apply alignment
    aligned_depth = scale * raw_depth + offset
    aligned_depth = np.clip(aligned_depth, 0.01, None)

    # Confidence: high at overlap, decays into gap regions
    from scipy.ndimage import distance_transform_edt
    dist_from_known = distance_transform_edt(gap_mask)
    confidence = np.exp(-dist_from_known / confidence_decay_pixels)
    confidence[~gap_mask] = 1.0

    return AlignedDepth(
        depth=aligned_depth,
        confidence=confidence,
        scale=float(scale),
        offset=float(offset),
    )
