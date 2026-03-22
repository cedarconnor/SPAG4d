"""RefineResult: output of the refinement pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RoundStats:
    """Per-round statistics."""
    round_number: int
    cameras_processed: int
    gaussians_seeded: int
    gaussians_promoted: int
    gaussians_pruned: int
    original_psnr: float
    elapsed_seconds: float


@dataclass
class RefineResult:
    """Output of the refinement pipeline."""
    output_path: Path
    original_gaussian_count: int
    final_gaussian_count: int
    gaussians_added: int
    gaussians_pruned: int
    original_psnr_before: float
    original_psnr_after: float
    rounds_completed: int
    round_stats: List[RoundStats] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    synthesis_backend_used: str = ""
    heatmap_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
