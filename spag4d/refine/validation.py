"""Multi-metric validation for Refine v2."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnchorCheckResult:
    passed: bool
    baseline_psnr: float
    current_psnr: float
    floor: float
    degradation: float


@dataclass
class ValidationReport:
    anchor_check: AnchorCheckResult
    coverage_before: float
    coverage_after: float
    coverage_improvement: float
    iteration: int


def compute_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute PSNR between two float images in [0, 1].

    Args:
        image_a: Reference image as float array, any shape.
        image_b: Comparison image, same shape as image_a.

    Returns:
        PSNR in dB, capped at 100.0 for effectively identical images.
    """
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_coverage(hole_masks: list) -> float:
    """Compute average coverage (non-hole fraction) across a list of masks.

    Args:
        hole_masks: List of 2-D float arrays where 1.0 = hole, 0.0 = filled.

    Returns:
        Average fraction of filled pixels in [0, 1]. Returns 1.0 for empty list.
    """
    if not hole_masks:
        return 1.0
    fractions = [float(1.0 - np.mean(np.asarray(m, dtype=np.float64))) for m in hole_masks]
    return float(np.mean(fractions))


def check_source_anchor(
    baseline_psnr: float,
    current_psnr: float,
    floor: float = 25.0,
    max_degradation: float = 1.0,
) -> AnchorCheckResult:
    """Gate whether refinement has degraded source quality too much.

    Args:
        baseline_psnr: PSNR of the unrefined output (reference).
        current_psnr: PSNR of the refined output being evaluated.
        floor: Minimum acceptable PSNR regardless of baseline (default 25 dB).
        max_degradation: Maximum allowed drop from baseline in dB (default 1.0).

    Returns:
        AnchorCheckResult with passed flag and diagnostic fields.
    """
    degradation = max(0.0, baseline_psnr - current_psnr)
    passed = (current_psnr >= floor) and (degradation <= max_degradation)
    if not passed:
        logger.warning(
            "Source anchor check FAILED: current_psnr=%.2f floor=%.2f "
            "baseline_psnr=%.2f degradation=%.2f max_degradation=%.2f",
            current_psnr,
            floor,
            baseline_psnr,
            degradation,
            max_degradation,
        )
    return AnchorCheckResult(
        passed=passed,
        baseline_psnr=baseline_psnr,
        current_psnr=current_psnr,
        floor=floor,
        degradation=degradation,
    )
