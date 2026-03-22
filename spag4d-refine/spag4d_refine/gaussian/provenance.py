"""Gaussian provenance tracking."""

from enum import IntEnum


class GaussianSource(IntEnum):
    """Provenance of each Gaussian in the cloud."""
    ORIGINAL = 0   # From the input PLY
    SEEDED = 1     # Shadow Gaussian from synthesis + depth
    PROMOTED = 2   # Validated and promoted from SEEDED
    PRUNED = 3     # Marked for removal (kept for bookkeeping)
