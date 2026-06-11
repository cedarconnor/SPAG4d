"""Splat refinement pipeline (OmniRoam v2 and geometric backends)."""

from .pipeline_v2 import refine_splat_v2, OmniRoamConfig
from .geometric import refine_splat_geometric, GeometricRefineConfig
from .artifixer3d_pipeline import refine_splat_artifixer3d
from .artifixer3d_config import ArtiFixer3DConfig

__all__ = [
    "refine_splat_v2", "OmniRoamConfig",
    "refine_splat_geometric", "GeometricRefineConfig",
    "refine_splat_artifixer3d", "ArtiFixer3DConfig",
]
