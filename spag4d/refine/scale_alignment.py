"""Scale alignment utilities for the GSFix3D refinement pipeline.

Provides two mechanisms for determining the scale factor between a splat's
internal coordinate space and the metric world frame implied by OmniRoam poses:

- parse_scale_config: interprets a user-provided config string ("none",
  "manual:<value>", or "reprojection").
- estimate_scale_factor: log-uniform grid search using cosine similarity
  between splat renders and a reference OmniRoam frame.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import numpy as np

__all__ = ["parse_scale_config", "estimate_scale_factor"]

# Sentinel used when "reprojection" mode is requested.
_REPROJECTION_SENTINEL = "reprojection"


def parse_scale_config(
    config_str: str,
) -> Optional[Union[float, str]]:
    """Parse a scale-config string into a Python value.

    Parameters
    ----------
    config_str:
        One of:
        - ``"none"``          -> ``None``  (pipeline uses unit scale)
        - ``"manual:<v>"``    -> ``float(v)``
        - ``"reprojection"``  -> the sentinel string ``"reprojection"``

    Returns
    -------
    None | float | str
        Parsed value; raises ``ValueError`` for unrecognised formats.
    """
    config_str = config_str.strip().lower()

    if config_str == "none":
        return None

    if config_str == "reprojection":
        return _REPROJECTION_SENTINEL

    if config_str.startswith("manual:"):
        raw = config_str[len("manual:"):]
        try:
            return float(raw)
        except ValueError:
            raise ValueError(
                f"Invalid manual scale value: {raw!r}. "
                "Expected format: \"manual:<float>\", e.g. \"manual:2.5\""
            )

    raise ValueError(
        f"Unrecognised scale config: {config_str!r}. "
        "Expected one of: \"none\", \"manual:<float>\", \"reprojection\""
    )


def estimate_scale_factor(
    render_fn: Callable[[float], np.ndarray],
    omniroam_frame1: np.ndarray,
    search_range: Tuple[float, float] = (0.1, 10.0),
    num_samples: int = 32,
    min_similarity: float = 0.3,
) -> float:
    """Find the scale factor that best aligns a splat render to a reference frame.

    Performs a log-uniform grid search over ``search_range``, rendering the
    splat at each candidate scale and comparing the result to
    ``omniroam_frame1`` via cosine similarity on the flattened pixel arrays.

    Parameters
    ----------
    render_fn:
        Callable ``scale -> (H, W, 3) float32 ndarray``.  In production this
        renders the Gaussian splat from an offset pose multiplied by *scale*;
        in tests it is a mock.
    omniroam_frame1:
        Reference image ``(H, W, 3) float32`` from the OmniRoam trajectory.
    search_range:
        ``(min_scale, max_scale)`` bounds for the grid search (log-uniform).
    num_samples:
        Number of candidate scales to evaluate.
    min_similarity:
        If the best cosine similarity found is below this threshold the method
        falls back to ``1.0`` (no scaling), indicating the search failed to
        find a meaningful alignment.

    Returns
    -------
    float
        Best-matching scale, or ``1.0`` if no candidate exceeded
        ``min_similarity``.
    """
    lo, hi = search_range
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(
            f"search_range must satisfy 0 < lo < hi, got ({lo}, {hi})"
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    scales = np.logspace(math.log10(lo), math.log10(hi), num_samples)

    # Flatten and normalise the reference frame once.
    ref_flat = omniroam_frame1.ravel().astype(np.float64)
    ref_norm = np.linalg.norm(ref_flat)

    best_scale = 1.0
    best_sim = -np.inf

    for scale in scales:
        rendered = render_fn(float(scale))
        cand_flat = np.asarray(rendered, dtype=np.float64).ravel()
        cand_norm = np.linalg.norm(cand_flat)

        if ref_norm < 1e-12 or cand_norm < 1e-12:
            sim = 0.0
        else:
            sim = float(np.dot(ref_flat, cand_flat) / (ref_norm * cand_norm))

        if sim > best_sim:
            best_sim = sim
            best_scale = float(scale)

    if best_sim < min_similarity:
        return 1.0

    return best_scale
