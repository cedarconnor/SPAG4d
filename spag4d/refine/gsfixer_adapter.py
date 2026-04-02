"""Phase 2: GSFixer model loading, fine-tuning, and inference."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Ensure GSFix3D is importable (it's a submodule, not an installed package).
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


class GSFixerAdapter:
    """Wraps GSFix3D's MarigoldGSFixerPipeline for SPAG-4D integration.

    Lifecycle: ``__init__`` -> ``load()`` -> ``finetune()`` (optional)
               -> ``infer()`` -> ``unload()``.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.pipe = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Load the pretrained GSFixer diffusion pipeline."""
        from marigold import MarigoldGSFixerPipeline

        logger.info("Loading GSFixer from %s ...", self.checkpoint_path)
        self.pipe = MarigoldGSFixerPipeline.from_pretrained(self.checkpoint_path)
        self.pipe = self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception:
            logger.info("xformers not available, using default attention")

        logger.info("GSFixer loaded successfully")

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        gs_renders: List[np.ndarray],
        gt_images: List[np.ndarray],
        mesh,
        cameras,
        train_steps: int = 500,
        learning_rate: float = 1e-5,
    ):
        """Fine-tune the GSFixer UNet on scene-specific training pairs.

        Uses cubemap GT views (extracted from the panorama) as ground truth
        and GS renders of those same views as conditioning input.  This
        teaches the model the visual style of *this specific scene* so that
        it produces consistent inpainted content later.

        Args:
            gs_renders: list of (H, W, 3) float32 numpy arrays in [0, 1].
            gt_images:  list of (H, W, 3) float32 numpy arrays in [0, 1].
            mesh:       trimesh.Trimesh (unused for single-input fine-tuning;
                        reserved for future dual-conditioning).
            cameras:    list of CameraPose objects (unused currently).
            train_steps: number of gradient steps.
            learning_rate: Adam learning rate.
        """
        if self.pipe is None:
            raise RuntimeError("Call load() before finetune()")

        logger.info("Fine-tuning GSFixer UNet for %d steps ...", train_steps)

        # Freeze everything except the UNet.
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        unet = self.pipe.unet
        unet.train()
        optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

        # Pre-encode empty text embedding (stored as self.pipe.empty_text_embed).
        self.pipe.encode_empty_text()

        n_pairs = len(gs_renders)
        if n_pairs == 0:
            logger.warning("No training pairs provided, skipping fine-tuning")
            unet.eval()
            return

        # Determine whether the UNet supports dual conditioning (12 input
        # channels) or single conditioning only (8 channels).
        unet_in_ch = unet.config.in_channels
        dual_input = unet_in_ch == 12
        logger.info(
            "UNet input channels: %d (%s conditioning)",
            unet_in_ch,
            "dual" if dual_input else "single",
        )

        for step in range(train_steps):
            idx = step % n_pairs

            # (H,W,3) float32 [0,1] -> (1,3,H,W) float32 [-1,1]
            gs_tensor = self._numpy_to_latent_input(gs_renders[idx])
            gt_tensor = self._numpy_to_latent_input(gt_images[idx])

            # Encode to latent space (deterministic: uses posterior mean).
            with torch.no_grad():
                gs_latent = self.pipe.encode_rgb(gs_tensor)
                gt_latent = self.pipe.encode_rgb(gt_tensor)

            # Random diffusion timestep.
            timestep = torch.randint(
                0,
                self.pipe.scheduler.config.num_train_timesteps,
                (1,),
                device=self.device,
            )

            # Forward diffusion: add noise to ground-truth latent.
            noise = torch.randn_like(gt_latent)
            noisy_latent = self.pipe.scheduler.add_noise(gt_latent, noise, timestep)

            # Build UNet input.  Match the concat order used by the pipeline:
            #   dual:   [mesh_latent, gs_latent, noisy_target_latent]  (12 ch)
            #   single: [gs_latent, noisy_target_latent]               (8 ch)
            if dual_input:
                # For now we duplicate the GS latent as the mesh channel;
                # a proper mesh render could be substituted later.
                unet_input = torch.cat([gs_latent, gs_latent, noisy_latent], dim=1)
            else:
                unet_input = torch.cat([gs_latent, noisy_latent], dim=1)

            # Expand empty text embedding to batch dimension.
            batch_text = self.pipe.empty_text_embed.repeat(
                (gs_latent.shape[0], 1, 1)
            ).to(self.device)

            # Predict noise residual.
            noise_pred = unet(
                unet_input, timestep, encoder_hidden_states=batch_text
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info(
                    "  finetune step %d/%d, loss=%.4f", step, train_steps, loss.item()
                )

        unet.eval()
        # Restore grad flags so VAE/text_encoder can be used normally later.
        self.pipe.vae.requires_grad_(True)
        self.pipe.text_encoder.requires_grad_(True)
        logger.info("Fine-tuning complete")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        gs_renders: List[np.ndarray],
        hole_masks: List[np.ndarray],
        mesh,
        cameras,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> List[np.ndarray]:
        """Run GSFixer inference on GS renders with holes.

        For each view the pipeline receives the GS render as primary
        conditioning (and optionally a mesh render as secondary
        conditioning).  The repaired output is composited with the
        original: unchanged pixels are kept intact and only hole regions
        are replaced by the model output.

        Args:
            gs_renders: list of (H, W, 3) float32 numpy arrays in [0, 1].
            hole_masks: list of (H, W) float32 binary masks (1 = hole).
            mesh:       trimesh.Trimesh for mesh conditioning renders
                        (may be ``None``).
            cameras:    list of CameraPose objects.
            num_steps:  DDIM denoising steps.
            guidance_scale: unused (pipeline has no text guidance).

        Returns:
            List of (H, W, 3) float32 repaired images in [0, 1].
        """
        if self.pipe is None:
            raise RuntimeError("Call load() before infer()")

        logger.info("Running GSFixer inference on %d views ...", len(gs_renders))
        repaired: List[np.ndarray] = []

        from .mesh_extract import render_mesh

        for i, (gs_img, mask, cam) in enumerate(
            zip(gs_renders, hole_masks, cameras)
        ):
            hole_frac = float(mask.mean())
            logger.info(
                "  Repairing view %d/%d (holes: %.1f%%)",
                i + 1,
                len(gs_renders),
                hole_frac * 100,
            )

            # Convert numpy [0,1] float to PIL uint8.
            gs_pil = Image.fromarray(
                (gs_img * 255).clip(0, 255).astype(np.uint8)
            )

            # Render mesh for dual conditioning (falls back to gray if None).
            mesh_img = render_mesh(
                mesh, cam, resolution=(gs_img.shape[0], gs_img.shape[1])
            )
            mesh_pil = Image.fromarray(
                (mesh_img * 255).clip(0, 255).astype(np.uint8)
            )

            try:
                output = self.pipe(
                    gs_pil,
                    condition_image2=mesh_pil if mesh is not None else None,
                    denoising_steps=num_steps,
                    processing_res=0,  # native resolution
                    match_input_res=True,
                    show_progress_bar=False,
                )
                result = (
                    np.array(output.fixed_rgb).astype(np.float32) / 255.0
                )
            except Exception as e:
                logger.warning(
                    "  GSFixer inference failed (%s), keeping original", e
                )
                result = gs_img.copy()

            # Soft composite: feather the mask boundary to avoid hard seams.
            # Gaussian blur the binary mask to create a soft transition.
            from scipy.ndimage import gaussian_filter
            soft_mask = gaussian_filter(mask.astype(np.float64), sigma=5.0)
            soft_mask = np.clip(soft_mask, 0, 1).astype(np.float32)
            mask_3d = soft_mask[..., None]
            composited = gs_img * (1.0 - mask_3d) + result * mask_3d
            repaired.append(composited.astype(np.float32))

        logger.info("GSFixer inference complete")
        return repaired

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self):
        """Free GPU memory held by the pipeline."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GSFixer unloaded, GPU memory freed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _numpy_to_latent_input(self, img_np: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) float32 [0, 1] to (1, 3, H, W) [-1, 1] on device."""
        t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        t = t.to(dtype=self.pipe.dtype, device=self.device)
        t = t * 2.0 - 1.0
        return t
