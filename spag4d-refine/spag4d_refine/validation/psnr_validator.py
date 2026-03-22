"""Original-view PSNR tracking to prevent quality degradation."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..camera.pinhole import PinholeCamera
from ..gaussian.cloud import GaussianCloud

logger = logging.getLogger(__name__)


def compute_psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute PSNR between two images.

    Args:
        image: [H, W, 3] float32 [0, 1]
        reference: [H, W, 3] float32 [0, 1]

    Returns:
        PSNR in dB
    """
    mse = np.mean((image - reference) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * math.log10(1.0 / mse))


def validate_original_view(
    cloud: GaussianCloud,
    original_view_camera: PinholeCamera,
    original_view_image: np.ndarray,
    device: str = "cuda",
) -> float:
    """
    Render the cloud from the original panorama position and measure PSNR.

    Args:
        cloud: Current GaussianCloud
        original_view_camera: Camera at panorama center
        original_view_image: [H, W, 3] float32 reference image

    Returns:
        PSNR in dB
    """
    from ..renderer.gsplat_renderer import GsplatRenderer

    renderer = GsplatRenderer(cloud, device=device)
    result = renderer.render(original_view_camera)

    psnr = compute_psnr(result.rgb, original_view_image)
    logger.info(f"Original-view PSNR: {psnr:.2f} dB")
    return psnr


def check_psnr_regression(
    psnr_before: float,
    psnr_after: float,
    max_drop: float = 0.5,
) -> bool:
    """
    Check if PSNR drop exceeds threshold.

    Returns True if acceptable, False if hard stop needed.
    """
    drop = psnr_before - psnr_after
    if drop > max_drop:
        logger.error(
            f"PSNR regression: {psnr_before:.2f} → {psnr_after:.2f} "
            f"(drop={drop:.2f} dB > threshold={max_drop:.2f} dB)"
        )
        return False
    return True
