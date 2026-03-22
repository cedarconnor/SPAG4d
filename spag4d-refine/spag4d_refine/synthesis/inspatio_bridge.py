"""InSpatio-World bridge — Track 2 video-consistent synthesis."""

from __future__ import annotations

import logging

import numpy as np

from ..config import RefineConfig

logger = logging.getLogger(__name__)


class InSpatioBridge:
    """
    Track 2: InSpatio-World video-consistent synthesis.

    Escalation path when per-frame Klein repair hits its limits.
    Uses trajectory-based multi-frame consistency.
    """

    def __init__(self, config: RefineConfig):
        self.config = config

    def repair(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "InSpatio-World bridge is not yet implemented. "
            "This is the Track 2 escalation path for trajectory-based synthesis. "
            "Use klein-sharp, flux-fill, or sdxl backends instead."
        )

    @property
    def backend_name(self) -> str:
        return "inspatio"
