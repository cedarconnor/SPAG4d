"""FLUX.1 Fill inpainting backend (fallback)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import RefineConfig

logger = logging.getLogger(__name__)


class FluxFillInpainter:
    """FLUX.1 Fill mask-based inpainting (no forward warp conditioning)."""

    def __init__(self, config: RefineConfig):
        self.config = config
        self._pipeline = None

    def _load_pipeline(self):
        import torch
        from diffusers import FluxFillPipeline

        device = torch.device(self.config.device)
        dtype = torch.float16 if self.config.mixed_precision else torch.float32

        logger.info(f"Loading FLUX Fill: {self.config.flux_fill_model}")
        self._pipeline = FluxFillPipeline.from_pretrained(
            self.config.flux_fill_model,
            torch_dtype=dtype,
        ).to(device)

    def repair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "Fill in the missing regions with consistent content",
    ) -> np.ndarray:
        """
        Inpaint masked regions.

        Args:
            image: [H, W, 3] float32 [0, 1]
            mask: [H, W] bool — True where content needs generation
            prompt: Text prompt

        Returns:
            [H, W, 3] float32 repaired image
        """
        if self._pipeline is None:
            self._load_pipeline()

        from PIL import Image as PILImage

        pil_image = PILImage.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
        pil_mask = PILImage.fromarray((mask.astype(np.uint8) * 255))

        result = self._pipeline(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=self.config.flux_fill_steps,
            guidance_scale=self.config.flux_fill_guidance_scale,
        )

        output = np.array(result.images[0]).astype(np.float32) / 255.0
        return output

    @property
    def backend_name(self) -> str:
        return "flux-fill"
