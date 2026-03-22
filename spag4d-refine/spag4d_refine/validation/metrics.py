"""Refinement success metrics: gap coverage, provenance breakdown, diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..gaussian.cloud import GaussianCloud
from ..gaussian.provenance import GaussianSource

logger = logging.getLogger(__name__)


@dataclass
class RefinementMetrics:
    """Full metrics for a refinement run."""
    # Gaussian counts
    original_count: int = 0
    final_count: int = 0
    seeded_count: int = 0
    promoted_count: int = 0
    pruned_count: int = 0

    # Gap coverage (per-round)
    gap_coverage_per_round: List[float] = field(default_factory=list)

    # PSNR
    original_psnr_before: float = 0.0
    original_psnr_after: float = 0.0
    psnr_drop: float = 0.0

    # Provenance breakdown
    provenance_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "original_count": self.original_count,
            "final_count": self.final_count,
            "gaussians_added": self.final_count - self.original_count,
            "seeded_count": self.seeded_count,
            "promoted_count": self.promoted_count,
            "pruned_count": self.pruned_count,
            "gap_coverage_per_round": self.gap_coverage_per_round,
            "original_psnr_before": round(self.original_psnr_before, 2),
            "original_psnr_after": round(self.original_psnr_after, 2),
            "psnr_drop": round(self.psnr_drop, 2),
            "provenance": self.provenance_counts,
        }


def compute_gap_coverage(
    region_map_before: np.ndarray,
    region_map_after: np.ndarray,
    gap_value: int = 3,
) -> float:
    """
    Compute fraction of gap pixels that were filled.

    Args:
        region_map_before: [H, W] region classification before refinement
        region_map_after: [H, W] region classification after refinement
        gap_value: Value representing TYPE_C (gap) in region_map

    Returns:
        Coverage ratio in [0, 1]. 1.0 = all gaps filled.
    """
    gaps_before = np.sum(region_map_before == gap_value)
    if gaps_before == 0:
        return 1.0
    gaps_after = np.sum(region_map_after == gap_value)
    return float(1.0 - gaps_after / gaps_before)


def compute_provenance_breakdown(cloud: GaussianCloud) -> Dict[str, int]:
    """Count Gaussians by provenance type."""
    counts = {}
    for source in GaussianSource:
        count = int(np.sum(cloud.provenance == source))
        if count > 0:
            counts[source.name] = count
    return counts


def save_gap_overlay(
    render_rgb: np.ndarray,
    gap_mask: np.ndarray,
    output_path: Path,
    color: tuple = (1.0, 0.0, 0.0),
    alpha: float = 0.4,
) -> None:
    """
    Save an image with gap regions highlighted in color.

    Args:
        render_rgb: [H, W, 3] float32 rendered image
        gap_mask: [H, W] bool — True where gaps exist
        output_path: Where to save
        color: RGB overlay color
        alpha: Overlay opacity
    """
    from PIL import Image

    overlay = render_rgb.copy()
    for c in range(3):
        overlay[gap_mask, c] = (
            overlay[gap_mask, c] * (1 - alpha) + color[c] * alpha
        )
    img = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))
    img.save(str(output_path))
    logger.info(f"Saved gap overlay to {output_path}")


def save_before_after(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    output_path: Path,
) -> None:
    """Save side-by-side before/after comparison."""
    from PIL import Image

    H = min(before_rgb.shape[0], after_rgb.shape[0])
    W = min(before_rgb.shape[1], after_rgb.shape[1])
    before = before_rgb[:H, :W]
    after = after_rgb[:H, :W]

    gap = np.ones((H, 4, 3), dtype=np.float32)
    combined = np.concatenate([before, gap, after], axis=1)
    img = Image.fromarray((np.clip(combined, 0, 1) * 255).astype(np.uint8))
    img.save(str(output_path))
    logger.info(f"Saved before/after to {output_path}")
