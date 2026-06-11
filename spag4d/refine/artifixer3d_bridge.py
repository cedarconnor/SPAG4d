"""SPAG-4D cloud -> COLMAP scene bridge for the ArtiFixer3D backend.

Ports experiments/artifixer_eval/_bridge_to_colmap.py into a tested module.
ArtiFixer consumes a COLMAP scene (``images/`` + ``sparse/0/*.bin``) and rebuilds
its *own* 3DGRUT reconstruction — it does not ingest a PLY. This module renders an
orbit of perspective views from the SPAG cloud and writes that COLMAP scene, plus
the anchor/novel split the distill step requires.
"""
import struct
from pathlib import Path

import numpy as np

# COLMAP camera model ids — OPENCV = fx, fy, cx, cy, k1, k2, p1, p2
OPENCV_MODEL_ID = 4
SH_C0 = 0.28209479177387814


def select_anchor_indices(hole_fracs, quantile: float = 0.34):
    """Anchors = lowest-parallax (lowest hole-fraction) views.

    distill needs >=1 novel (non-anchor) view, so the anchor set is clamped
    to [1, n-1] views.
    """
    n = len(hole_fracs)
    order = sorted(range(n), key=lambda i: hole_fracs[i])
    k = max(1, min(n - 1, int(round(quantile * n))))
    return sorted(order[:k])
