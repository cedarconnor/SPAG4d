"""
Klein 9B + ml-sharp LoRA synthesis backend (default).

ml-sharp LoRA conditioning (3 images):
  Image 1: Forward-warped RGB — the reference scene with parallax
  Image 2: Broken splat render — the novel view with gaps/artifacts
  Image 3: Depth/disparity visualization — spatial context for repair

Prompt format (from LoRA model card):
  "Referring to the scene in image 1, restore the perspective of
   the scene in image 2. Repair the perspective and missing areas.
   The camera has moved by: {JSON}"

Graceful degradation:
  1. Klein + ml-sharp LoRA (best)
  2. Klein base without LoRA
  3. Clear error with instructions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..config import RefineConfig
from .panorama_conditioner import (
    build_repair_prompt,
    compute_relative_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class RepairInput:
    """Input for Klein repair."""
    forward_warped_rgb: np.ndarray    # [H, W, 3] float32 — image 1 (reference)
    broken_splat_render: np.ndarray   # [H, W, 3] float32 — image 2 (to repair)
    depth_disparity_vis: np.ndarray   # [H, W, 3] float32 — depth context
    ref_c2w: np.ndarray              # [4, 4]
    novel_c2w: np.ndarray            # [4, 4]
    gap_mask: np.ndarray             # [H, W] bool — regions to repair


def _convert_bfl_lora_to_diffusers(state_dict: Dict) -> Dict:
    """
    Convert ml-sharp LoRA keys from BFL naming to diffusers naming.

    BFL (original Flux):            Diffusers:
      double_blocks.N.img_attn.qkv  -> transformer.transformer_blocks.N.attn.to_q/k/v  (split)
      double_blocks.N.img_attn.proj -> transformer.transformer_blocks.N.attn.to_out.0
      double_blocks.N.img_mlp.0     -> transformer.transformer_blocks.N.ff.linear_in
      double_blocks.N.img_mlp.2     -> transformer.transformer_blocks.N.ff.linear_out
      double_blocks.N.txt_attn.qkv  -> transformer.transformer_blocks.N.attn.add_q_proj/... (split)
      double_blocks.N.txt_attn.proj -> transformer.transformer_blocks.N.attn.to_add_out
      double_blocks.N.txt_mlp.0     -> transformer.transformer_blocks.N.ff_context.linear_in
      double_blocks.N.txt_mlp.2     -> transformer.transformer_blocks.N.ff_context.linear_out
      single_blocks.N.linear1       -> transformer.single_transformer_blocks.N.attn.to_qkv_mlp_proj
      single_blocks.N.linear2       -> transformer.single_transformer_blocks.N.attn.to_out

    For QKV LoRA layers, the LoRA targets the fused QKV weight. Diffusers
    splits Q/K/V into separate modules, so we need to map the fused LoRA
    to each split target.
    """
    converted = {}

    # Simple 1:1 renames (no QKV splitting needed for LoRA — the LoRA
    # matrices target the fused weight and diffusers will handle the
    # shape matching during injection)
    rename_map = {
        # Double block image stream
        "img_attn.qkv": "attn.to_qkv",        # fused QKV
        "img_attn.proj": "attn.to_out.0",
        "img_mlp.0": "ff.linear_in",
        "img_mlp.2": "ff.linear_out",
        # Double block text stream
        "txt_attn.qkv": "attn.add_qkv_proj",   # fused QKV
        "txt_attn.proj": "attn.to_add_out",
        "txt_mlp.0": "ff_context.linear_in",
        "txt_mlp.2": "ff_context.linear_out",
    }

    for key, tensor in state_dict.items():
        new_key = key

        # double_blocks.N.xxx -> transformer.transformer_blocks.N.xxx
        if key.startswith("double_blocks."):
            m = re.match(r"double_blocks\.(\d+)\.(.*)", key)
            if m:
                block_idx = m.group(1)
                remainder = m.group(2)
                # Apply renames
                for old, new in rename_map.items():
                    if remainder.startswith(old):
                        remainder = remainder.replace(old, new, 1)
                        break
                new_key = f"transformer.transformer_blocks.{block_idx}.{remainder}"

        # single_blocks.N.xxx -> transformer.single_transformer_blocks.N.xxx
        elif key.startswith("single_blocks."):
            m = re.match(r"single_blocks\.(\d+)\.(.*)", key)
            if m:
                block_idx = m.group(1)
                remainder = m.group(2)
                # linear1 -> attn.to_qkv_mlp_proj, linear2 -> attn.to_out
                remainder = remainder.replace("linear1", "attn.to_qkv_mlp_proj", 1)
                remainder = remainder.replace("linear2", "attn.to_out", 1)
                new_key = f"transformer.single_transformer_blocks.{block_idx}.{remainder}"

        converted[new_key] = tensor

    logger.info(f"Converted {len(converted)} LoRA keys (BFL -> diffusers)")
    return converted


class KleinSharpRepairer:
    """Klein 9B + ml-sharp LoRA synthesis."""

    def __init__(self, config: RefineConfig):
        self.config = config
        self._pipeline = None
        self._backend_name = "unknown"

    def _load_pipeline(self):
        """Load Klein pipeline with ml-sharp LoRA."""
        import torch

        device = torch.device(self.config.device)
        dtype = torch.float16 if self.config.mixed_precision else torch.float32

        # Try KV-cache variant first, fall back to standard Klein
        PipeClass = None
        model_id = None

        if self.config.klein_use_kv_cache:
            try:
                from diffusers import Flux2KleinKVPipeline as _KV
                PipeClass = _KV
                model_id = self.config.klein_kv_model
                logger.info("Using Klein KV-cache pipeline")
            except ImportError:
                logger.info("Flux2KleinKVPipeline not available, trying standard Klein")

        if PipeClass is None:
            try:
                from diffusers import Flux2KleinPipeline as _Std
                PipeClass = _Std
                model_id = self.config.klein_base_model
                logger.info("Using standard Klein pipeline")
            except ImportError:
                raise RuntimeError(
                    "Klein pipeline not available. Install diffusers>=0.37.0.\n"
                    f"  Required model: {self.config.klein_base_model}\n"
                    f"  LoRA: {self.config.klein_sharp_lora}"
                )

        logger.info(f"Loading Klein pipeline: {model_id}")
        self._pipeline = PipeClass.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self._pipeline.enable_model_cpu_offload()
        logger.info("Klein pipeline loaded with CPU offload")

        # Load ml-sharp LoRA with BFL->diffusers key conversion
        try:
            self._load_lora_with_conversion()
            self._backend_name = "klein-sharp-lora"
            logger.info("Loaded ml-sharp LoRA successfully")
        except Exception as e:
            logger.warning(f"ml-sharp LoRA not available: {e}. Using Klein base.")
            self._backend_name = "klein-base"

    def _load_lora_with_conversion(self):
        """Load LoRA weights, converting BFL key names to diffusers format."""
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download

        lora_path = hf_hub_download(
            self.config.klein_sharp_lora,
            "flux2-klein9b-lora-mlsharp-3d-repair.safetensors",
        )

        # Load raw state dict
        raw_sd = {}
        with safe_open(lora_path, framework="pt") as f:
            for key in f.keys():
                raw_sd[key] = f.get_tensor(key)

        # Convert key names
        converted_sd = _convert_bfl_lora_to_diffusers(raw_sd)

        # Try loading with diffusers native method first
        try:
            self._pipeline.load_lora_weights(
                converted_sd,
                adapter_name="mlsharp",
            )
            return
        except Exception as e:
            logger.info(f"Native LoRA load failed ({e}), trying manual injection")

        # Manual injection fallback: apply LoRA weights directly to model
        self._inject_lora_manual(converted_sd)

    def _inject_lora_manual(self, lora_sd: Dict):
        """Manually inject LoRA weights into transformer modules (W' = W + B @ A)."""
        import torch

        transformer = self._pipeline.transformer
        injected = 0

        # Group by target module
        lora_pairs = {}
        for key, tensor in lora_sd.items():
            # Strip 'transformer.' prefix
            if key.startswith("transformer."):
                key = key[len("transformer."):]
            # Parse: module_path.lora_A.weight or module_path.lora_B.weight
            m = re.match(r"(.+)\.(lora_[AB])\.weight", key)
            if m:
                module_path = m.group(1)
                ab = m.group(2)
                lora_pairs.setdefault(module_path, {})[ab] = tensor

        for module_path, matrices in lora_pairs.items():
            if "lora_A" not in matrices or "lora_B" not in matrices:
                continue

            # Find the target module
            parts = module_path.split(".")
            target = transformer
            try:
                for p in parts:
                    if p.isdigit():
                        target = target[int(p)]
                    else:
                        target = getattr(target, p)
            except (AttributeError, IndexError, KeyError):
                logger.debug(f"Module not found: {module_path}")
                continue

            if not hasattr(target, "weight"):
                continue

            # Apply LoRA: W' = W + B @ A
            A = matrices["lora_A"].to(target.weight.dtype).to(target.weight.device)
            B = matrices["lora_B"].to(target.weight.dtype).to(target.weight.device)
            target.weight.data += B @ A
            injected += 1

        logger.info(f"Manually injected LoRA into {injected} modules")

    def repair(self, inputs: RepairInput) -> np.ndarray:
        """
        Repair disoccluded regions using Klein + ml-sharp LoRA.

        Args:
            inputs: RepairInput with reference + broken images and camera transforms.

        Returns:
            [H, W, 3] float32 repaired RGB image.
        """
        if self._pipeline is None:
            self._load_pipeline()

        from PIL import Image

        # Compute relative transform and build ml-sharp prompt
        transform = compute_relative_transform(inputs.ref_c2w, inputs.novel_c2w)
        prompt = build_repair_prompt(transform)
        logger.info(f"Prompt: {prompt}")

        # ml-sharp LoRA conditioning (3 images per v6 spec):
        #   Image 1: reference scene (forward-warped RGB with parallax)
        #   Image 2: broken novel view (splat render with gaps)
        #   Image 3: depth/disparity reference for spatial context
        img1 = self._to_pil(inputs.forward_warped_rgb, size=(1024, 1024))
        img2 = self._to_pil(inputs.broken_splat_render, size=(1024, 1024))
        img3 = self._to_pil(inputs.depth_disparity_vis, size=(1024, 1024))

        result = self._pipeline(
            prompt=prompt,
            image=[img1, img2, img3],
            height=1024,
            width=1024,
            num_inference_steps=self.config.klein_num_steps,
            guidance_scale=self.config.klein_guidance_scale,
        )

        output = np.array(result.images[0]).astype(np.float32) / 255.0

        # Resize to match gap_mask if needed
        if output.shape[:2] != inputs.gap_mask.shape:
            pil_out = Image.fromarray((output * 255).astype(np.uint8))
            pil_out = pil_out.resize(
                (inputs.gap_mask.shape[1], inputs.gap_mask.shape[0]), Image.LANCZOS
            )
            output = np.array(pil_out).astype(np.float32) / 255.0

        return output

    @staticmethod
    def _to_pil(
        img: np.ndarray,
        size: tuple[int, int] = (1024, 1024),
    ):
        """Convert float32 [H,W,3] to PIL Image, resized to target."""
        from PIL import Image
        uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(uint8)
        if pil.size != size:
            pil = pil.resize(size, Image.LANCZOS)
        return pil

    @property
    def backend_name(self) -> str:
        return self._backend_name
