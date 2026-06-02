# spag4d/refine/geometric/diagnostics.py
"""Structured diagnostics for the geometric refine pipeline."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from .depth_align import AlignmentResult


@dataclass
class FrameDiagnostics:
    frame_idx: int
    alignment: AlignmentResult
    num_candidates: int
    num_kept_alpha: int
    num_kept_disoccl: int
    num_candidates_pre_consistency: int
    num_candidates_post_consistency: int
    consistency_drop_rate: float
    avg_support_count: float
    skipped: bool
    skip_reason: str | None
    wall_time_seconds: float


@dataclass
class PipelineDiagnostics:
    total_frames: int
    frames_skipped: int
    total_new_gaussians: int
    trajectory_coverage_warning: bool
    scale_consistency_std_ratio: float
    color_polish_seam_delta: float | None
    total_runtime_seconds: float


def write_diagnostics_json(
    frame_diags: List[FrameDiagnostics],
    pipeline_diag: PipelineDiagnostics,
    output_path: Path,
) -> None:
    """Write diagnostics to JSON alongside the output PLY."""
    data = {
        "pipeline": asdict(pipeline_diag),
        "frames": [asdict(f) for f in frame_diags],
    }
    output_path.write_text(json.dumps(data, indent=2, default=str))
