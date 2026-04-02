"""PLY format conversion between SPAG-4D and GSFix3D."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_gaussians_from_ply(ply_path: str, device: str = "cuda"):
    """Load a SPAG-4D PLY file into GSFix3D's GaussianModel format."""
    logger.info(f"[stub] load_gaussians_from_ply({ply_path})")
    return None


def save_gaussians_to_ply(gaussians, output_path: str):
    """Save a GaussianModel back to SPAG-4D PLY format."""
    logger.info(f"[stub] save_gaussians_to_ply -> {output_path}")
