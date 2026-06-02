"""Splat refinement pipeline (OmniRoam v2 and geometric backends)."""

from .pipeline_v2 import refine_splat_v2, OmniRoamConfig
from .geometric import refine_splat_geometric, GeometricRefineConfig

__all__ = ["refine_splat_v2", "OmniRoamConfig", "refine_splat_geometric", "GeometricRefineConfig"]
