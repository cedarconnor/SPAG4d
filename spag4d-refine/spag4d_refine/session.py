"""Per-round refinement session state tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .result import RoundStats


@dataclass
class RefineSession:
    """Tracks state across refinement rounds."""
    round_number: int = 0
    round_stats: List[RoundStats] = field(default_factory=list)
    total_start_time: float = field(default_factory=time.time)
    round_start_time: float = 0.0
    intermediate_ply_paths: List[Path] = field(default_factory=list)

    def begin_round(self) -> None:
        """Start a new refinement round."""
        self.round_number += 1
        self.round_start_time = time.time()

    def end_round(
        self,
        cameras_processed: int,
        gaussians_seeded: int,
        gaussians_promoted: int,
        gaussians_pruned: int,
        original_psnr: float,
    ) -> RoundStats:
        """Finish a round and record stats."""
        elapsed = time.time() - self.round_start_time
        stats = RoundStats(
            round_number=self.round_number,
            cameras_processed=cameras_processed,
            gaussians_seeded=gaussians_seeded,
            gaussians_promoted=gaussians_promoted,
            gaussians_pruned=gaussians_pruned,
            original_psnr=original_psnr,
            elapsed_seconds=elapsed,
        )
        self.round_stats.append(stats)
        return stats

    @property
    def total_elapsed(self) -> float:
        return time.time() - self.total_start_time
