"""Phase 2: GSFixer model loading, fine-tuning, and inference."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class GSFixerAdapter:
    """Wraps GSFix3D's MarigoldGSFixerPipeline for SPAG-4D integration."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        logger.info(f"[stub] GSFixerAdapter created (checkpoint={checkpoint_path})")

    def load(self):
        logger.info("[stub] GSFixerAdapter.load()")

    def finetune(self, gs_renders, gt_images, mesh, cameras,
                 train_steps=500, learning_rate=1e-5):
        logger.info(f"[stub] GSFixerAdapter.finetune({train_steps} steps)")

    def infer(self, gs_renders, hole_masks, mesh, cameras,
              num_steps=50, guidance_scale=7.5):
        logger.info(f"[stub] GSFixerAdapter.infer({len(gs_renders)} views)")
        repaired = []
        for img, mask in zip(gs_renders, hole_masks):
            out = img.copy()
            if mask.sum() > 0:
                avg_color = img[mask < 0.5].mean(axis=0) if (mask < 0.5).any() else np.array([0.5, 0.5, 0.5])
                out[mask > 0.5] = avg_color
            repaired.append(out)
        return repaired

    def unload(self):
        self.model = None
        logger.info("[stub] GSFixerAdapter.unload()")
