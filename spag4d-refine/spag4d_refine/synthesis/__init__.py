"""Synthesis backend dispatch."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RefineConfig


def get_synthesis_backend(config: "RefineConfig"):
    """Route to the appropriate synthesis backend based on config."""
    backend = config.synthesis_backend

    if backend == "klein-sharp":
        from .klein_sharp_repair import KleinSharpRepairer
        return KleinSharpRepairer(config)
    elif backend == "flux-fill":
        from .flux_fill_inpainter import FluxFillInpainter
        return FluxFillInpainter(config)
    elif backend == "sdxl":
        from .sdxl_inpainter import SDXLInpainter
        return SDXLInpainter(config)
    elif backend == "inspatio":
        from .inspatio_bridge import InSpatioBridge
        return InSpatioBridge(config)
    else:
        raise ValueError(f"Unknown synthesis backend: {backend}")
