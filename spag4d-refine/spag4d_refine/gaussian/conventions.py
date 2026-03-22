"""PLY convention auto-detection for Gaussian splats."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ConventionFlags:
    """Detected PLY conventions."""
    quat_order: Literal["WXYZ", "XYZW"]
    scale_encoding: Literal["log", "linear"]
    opacity_encoding: Literal["logit", "raw"]


def detect_conventions(vertex_data: np.ndarray) -> ConventionFlags:
    """
    Auto-detect quaternion order, scale encoding, and opacity encoding
    from raw PLY vertex data.

    Heuristics:
    - Quaternion: If rot_0 values cluster near 1.0, it's W-first (WXYZ).
    - Scale: If scale values are mostly negative, they're log-encoded.
    - Opacity: If opacity values span beyond [0,1], they're logit-encoded.
    """
    # Quaternion order detection
    # Standard 3DGS PLY format is WXYZ. For real scenes, rot_0 (=W) has the
    # largest mean absolute value among the four components. We compare rot_0
    # against the other components rather than using an absolute threshold,
    # since random orientations can have low mean |W|.
    rot_0 = vertex_data["rot_0"]
    rot_1 = vertex_data["rot_1"]
    rot_2 = vertex_data["rot_2"]
    rot_3 = vertex_data["rot_3"]

    mean_abs = [np.mean(np.abs(r)) for r in [rot_0, rot_1, rot_2, rot_3]]
    # WXYZ: rot_0 is W (largest for small rotations). Default to WXYZ (standard).
    quat_order: Literal["WXYZ", "XYZW"] = "WXYZ"
    # Only switch to XYZW if rot_3 (would-be W) is clearly larger than rot_0
    if mean_abs[3] > mean_abs[0] * 1.5:
        quat_order = "XYZW"

    # Scale encoding detection
    scale_0 = vertex_data["scale_0"]
    # Log-encoded scales are typically negative (log of small values)
    frac_negative = np.mean(scale_0 < 0)
    scale_encoding: Literal["log", "linear"] = "log" if frac_negative > 0.3 else "linear"

    # Opacity encoding detection
    opacity = vertex_data["opacity"]
    # Logit-encoded opacities can be any real number (not bounded to [0,1])
    in_01 = np.mean((opacity >= -0.01) & (opacity <= 1.01))
    opacity_encoding: Literal["logit", "raw"] = "raw" if in_01 > 0.95 else "logit"

    return ConventionFlags(
        quat_order=quat_order,
        scale_encoding=scale_encoding,
        opacity_encoding=opacity_encoding,
    )
