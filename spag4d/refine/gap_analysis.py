"""Gap analysis: render splat from evaluation viewpoints and classify gap regions."""

import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Map direction name to (start_deg, end_deg) azimuth range (exclusive end).
# "forward" wraps around 0 (315-360 and 0-45).
_DIRECTION_MAP = {
    "forward":  (315, 45),   # wraps around 0
    "right":    (45, 135),
    "backward": (135, 225),
    "left":     (225, 315),
}

# Ordered list used for stable iteration and sorting.
_DIRECTION_ORDER = ["forward", "right", "backward", "left"]


@dataclass
class GapReport:
    """Result of gap analysis across evaluation viewpoints."""
    avg_hole_fraction: float
    per_direction_fractions: dict
    worst_direction: str
    recommended_trajectories: list
    converged: bool


def _azimuth_to_direction(azimuth_deg: float) -> str:
    """Map an azimuth angle (0–360) to the nearest cardinal direction name.

    "forward" covers 315–360 and 0–45 (wraps around north).
    """
    az = azimuth_deg % 360.0

    # Check forward first — it wraps around 0.
    fwd_start, fwd_end = _DIRECTION_MAP["forward"]  # 315, 45
    if az >= fwd_start or az < fwd_end:
        return "forward"

    for name in ("right", "backward", "left"):
        start, end = _DIRECTION_MAP[name]
        if start <= az < end:
            return name

    # Fallback (should not be reached for valid azimuths).
    return "forward"


def classify_gap_directions(
    hole_masks,
    azimuths_deg,
    min_hole_fraction: float = 0.05,
) -> GapReport:
    """Classify gap severity by angular direction and build a GapReport.

    Args:
        hole_masks: Sequence of 2-D float arrays (H, W).  Values should be in
            [0, 1] where higher means more hole/gap.  The mean of each mask is
            used as the hole fraction for that viewpoint.
        azimuths_deg: Sequence of floats, one per mask, giving the camera
            azimuth of that viewpoint in degrees (0–360).
        min_hole_fraction: Convergence threshold.  If the overall average hole
            fraction is below this value, ``converged`` is set to True and no
            trajectories are recommended.

    Returns:
        GapReport with per-direction fractions and recommended trajectories
        sorted by severity (worst direction first).
    """
    if len(hole_masks) == 0:
        return GapReport(
            avg_hole_fraction=0.0,
            per_direction_fractions={d: 0.0 for d in _DIRECTION_ORDER},
            worst_direction=_DIRECTION_ORDER[0],
            recommended_trajectories=[],
            converged=True,
        )

    # Accumulate fractions per direction.
    direction_buckets: dict[str, list[float]] = {d: [] for d in _DIRECTION_ORDER}

    all_fractions: list[float] = []
    for mask, az in zip(hole_masks, azimuths_deg):
        fraction = float(np.mean(mask))
        all_fractions.append(fraction)
        direction = _azimuth_to_direction(float(az))
        direction_buckets[direction].append(fraction)

    avg_hole_fraction = float(np.mean(all_fractions))

    # Average within each direction (0.0 for directions with no viewpoints).
    per_direction_fractions: dict[str, float] = {}
    for d in _DIRECTION_ORDER:
        buckets = direction_buckets[d]
        per_direction_fractions[d] = float(np.mean(buckets)) if buckets else 0.0

    # Worst direction = highest average fraction.
    worst_direction = max(_DIRECTION_ORDER, key=lambda d: per_direction_fractions[d])

    # Recommend trajectories for directions exceeding the threshold, sorted by
    # severity descending.
    if avg_hole_fraction < min_hole_fraction:
        recommended_trajectories: list[str] = []
        converged = True
    else:
        recommended_trajectories = [
            d for d in sorted(
                _DIRECTION_ORDER,
                key=lambda d: per_direction_fractions[d],
                reverse=True,
            )
            if per_direction_fractions[d] >= min_hole_fraction
        ]
        converged = False

    return GapReport(
        avg_hole_fraction=avg_hole_fraction,
        per_direction_fractions=per_direction_fractions,
        worst_direction=worst_direction,
        recommended_trajectories=recommended_trajectories,
        converged=converged,
    )


def select_trajectories(report: GapReport, config) -> list:
    """Select OmniRoam trajectory presets based on the gap report and config.

    Args:
        report: A GapReport produced by :func:`classify_gap_directions`.
        config: An OmniRoamConfig instance.  Consults ``trajectory_mode`` and
            ``available_presets``.

    Returns:
        List of trajectory preset name strings.
    """
    mode = config.trajectory_mode

    if isinstance(mode, list):
        return mode

    if mode == "all":
        return list(config.available_presets)

    if mode == "auto":
        return list(report.recommended_trajectories)

    # Treat as a single preset name (e.g. "forward")
    return [mode]
