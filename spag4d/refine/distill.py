"""Phase 3: 3DGS distillation."""
import logging

logger = logging.getLogger(__name__)


def distill_to_gaussians(
    gaussians, repaired_images, cameras, hole_masks,
    original_images=None, original_cameras=None,
    num_iterations=3000, densify_interval=100,
    densify_grad_threshold=0.0002, **kwargs,
):
    """Optimize 3DGS to match repaired images. STUB: returns gaussians unchanged."""
    logger.info(f"[stub] distill_to_gaussians({num_iterations} iters, {len(repaired_images)} views)")
    return gaussians
