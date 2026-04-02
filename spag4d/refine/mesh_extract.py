"""Mesh extraction from depth map for GSFixer dual conditioning."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_conditioning_mesh(depth_map, panorama, simplify_ratio=0.1):
    """Extract a rough textured mesh from DAP depth. STUB: returns None."""
    logger.info(f"[stub] extract_conditioning_mesh (simplify={simplify_ratio})")
    return None


def render_mesh(mesh, camera, resolution=(512, 512)):
    """Render mesh from a camera pose. STUB: returns gray image."""
    logger.info("[stub] render_mesh")
    return np.ones((*resolution, 3), dtype=np.float32) * 0.5
