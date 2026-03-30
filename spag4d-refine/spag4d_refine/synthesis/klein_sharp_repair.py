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

    BFL (original Flux):            Diffusers (Flux2Transformer2DModel):
      double_blocks.N.img_attn.qkv  -> SPLIT into attn.to_q / to_k / to_v
      double_blocks.N.img_attn.proj -> transformer_blocks.N.attn.to_out.0
      double_blocks.N.img_mlp.0     -> transformer_blocks.N.ff.linear_in
      double_blocks.N.img_mlp.2     -> transformer_blocks.N.ff.linear_out
      double_blocks.N.txt_attn.qkv  -> SPLIT into attn.add_q_proj / add_k_proj / add_v_proj
      double_blocks.N.txt_attn.proj -> transformer_blocks.N.attn.to_add_out
      double_blocks.N.txt_mlp.0     -> transformer_blocks.N.ff_context.linear_in
      double_blocks.N.txt_mlp.2     -> transformer_blocks.N.ff_context.linear_out
      single_blocks.N.linear1       -> single_transformer_blocks.N.attn.to_qkv_mlp_proj
      single_blocks.N.linear2       -> single_transformer_blocks.N.attn.to_out

    CRITICAL: Double-block QKV LoRA targets a fused [3*hidden, in] weight but
    diffusers splits Q/K/V into separate modules. We must keep the raw A/B
    tensors and handle splitting during injection (see _inject_lora_manual).
    """
    converted = {}

    # 1:1 renames for non-fused layers
    rename_map = {
        # Double block image stream (non-QKV)
        "img_attn.proj": "attn.to_out.0",
        "img_mlp.0": "ff.linear_in",
        "img_mlp.2": "ff.linear_out",
        # Double block text stream (non-QKV)
        "txt_attn.proj": "attn.to_add_out",
        "txt_mlp.0": "ff_context.linear_in",
        "txt_mlp.2": "ff_context.linear_out",
    }

    # QKV fused keys → mark with __FUSED_QKV__ prefix for special handling
    # during injection. These get split into 3 separate module targets.
    qkv_fused_map = {
        "img_attn.qkv": "__FUSED_IMG_QKV__",
        "txt_attn.qkv": "__FUSED_TXT_QKV__",
    }

    for key, tensor in state_dict.items():
        new_key = key

        if key.startswith("double_blocks."):
            m = re.match(r"double_blocks\.(\d+)\.(.*)", key)
            if m:
                block_idx = m.group(1)
                remainder = m.group(2)

                # Check for fused QKV first
                is_fused = False
                for old, marker in qkv_fused_map.items():
                    if remainder.startswith(old):
                        suffix = remainder[len(old):]  # e.g. ".lora_A.weight"
                        new_key = f"{marker}.transformer_blocks.{block_idx}{suffix}"
                        is_fused = True
                        break

                if not is_fused:
                    for old, new in rename_map.items():
                        if remainder.startswith(old):
                            remainder = remainder.replace(old, new, 1)
                            break
                    new_key = f"transformer.transformer_blocks.{block_idx}.{remainder}"

        elif key.startswith("single_blocks."):
            m = re.match(r"single_blocks\.(\d+)\.(.*)", key)
            if m:
                block_idx = m.group(1)
                remainder = m.group(2)
                remainder = remainder.replace("linear1", "attn.to_qkv_mlp_proj", 1)
                remainder = remainder.replace("linear2", "attn.to_out", 1)
                new_key = f"transformer.single_transformer_blocks.{block_idx}.{remainder}"

        converted[new_key] = tensor

    n_fused = sum(1 for k in converted if k.startswith("__FUSED_"))
    n_direct = len(converted) - n_fused
    logger.info(
        f"Converted {len(converted)} LoRA keys (BFL -> diffusers): "
        f"{n_direct} direct, {n_fused} fused-QKV (will be split during injection)"
    )
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

        # Optionally load ml-sharp LoRA
        if self.config.klein_use_lora:
            try:
                self._load_lora_with_conversion()
                self._backend_name = "klein-sharp-lora"
                logger.info("Loaded ml-sharp LoRA successfully")
            except Exception as e:
                logger.warning(f"ml-sharp LoRA not available: {e}. Using Klein base.")
                self._backend_name = "klein-base"
        else:
            logger.info("Using Klein base (LoRA disabled)")
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

        # Use manual injection directly — native load_lora_weights with
        # fused QKV keys causes silent corruption and CUDA segfaults
        self._inject_lora_manual(converted_sd)

    def _inject_lora_manual(self, lora_sd: Dict):
        """
        Manually inject LoRA weights into transformer modules.

        Handles two cases:
        1. Direct injection (W' = W + B @ A) for 1:1 mapped modules
        2. Fused QKV splitting: compute B @ A, split along dim 0 into
           3 equal chunks, apply each to separate Q/K/V modules
        """
        import torch

        transformer = self._pipeline.transformer
        injected = 0
        skipped = 0

        # Separate fused-QKV keys from direct keys
        fused_pairs = {}   # marker -> {block_path -> {lora_A, lora_B}}
        direct_pairs = {}  # module_path -> {lora_A, lora_B}

        for key, tensor in lora_sd.items():
            # Check for fused QKV markers from _convert_bfl_lora_to_diffusers
            if key.startswith("__FUSED_"):
                m = re.match(r"(__FUSED_\w+__)\.(.+)\.(lora_[AB])\.weight", key)
                if m:
                    marker = m.group(1)
                    block_path = m.group(2)
                    ab = m.group(3)
                    fused_key = f"{marker}.{block_path}"
                    fused_pairs.setdefault(fused_key, {})[ab] = tensor
                continue

            # Direct (non-fused) keys
            clean = key
            if clean.startswith("transformer."):
                clean = clean[len("transformer."):]
            m = re.match(r"(.+)\.(lora_[AB])\.weight", clean)
            if m:
                module_path = m.group(1)
                ab = m.group(2)
                direct_pairs.setdefault(module_path, {})[ab] = tensor

        # --- Direct injection ---
        for module_path, matrices in direct_pairs.items():
            if "lora_A" not in matrices or "lora_B" not in matrices:
                continue

            target = self._resolve_module(transformer, module_path)
            if target is None or not hasattr(target, "weight"):
                skipped += 1
                continue

            delta = self._compute_delta(matrices, target.weight)
            target.weight.data += delta
            injected += 1

        # --- Fused QKV injection (split into separate Q, K, V) ---
        # Map fused markers to their 3 target module suffixes
        fused_targets = {
            "__FUSED_IMG_QKV__": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "__FUSED_TXT_QKV__": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
        }

        for fused_key, matrices in fused_pairs.items():
            if "lora_A" not in matrices or "lora_B" not in matrices:
                continue

            # Parse: __FUSED_IMG_QKV__.transformer_blocks.N
            m = re.match(r"(__FUSED_\w+__)\.(.+)", fused_key)
            if not m:
                continue
            marker = m.group(1)
            block_path = m.group(2)
            target_suffixes = fused_targets.get(marker)
            if not target_suffixes:
                continue

            # Find the first target to get dtype/device
            first_target = self._resolve_module(transformer, f"{block_path}.{target_suffixes[0]}")
            if first_target is None or not hasattr(first_target, "weight"):
                skipped += 1
                logger.debug(f"Fused QKV target not found: {block_path}.{target_suffixes[0]}")
                continue

            # Compute full fused delta: [3*hidden, in_features]
            compute_device = "cuda" if torch.cuda.is_available() else first_target.weight.device
            A = matrices["lora_A"].to(first_target.weight.dtype).to(compute_device)
            B = matrices["lora_B"].to(first_target.weight.dtype).to(compute_device)
            fused_delta = (B @ A)  # [3*hidden, in_features]

            # Split into 3 equal chunks along dim 0
            chunk_size = fused_delta.shape[0] // 3
            chunks = fused_delta.split(chunk_size, dim=0)
            if len(chunks) != 3:
                logger.warning(f"Fused QKV split failed: got {len(chunks)} chunks for {fused_key}")
                skipped += 1
                continue

            for suffix, chunk in zip(target_suffixes, chunks):
                target = self._resolve_module(transformer, f"{block_path}.{suffix}")
                if target is None or not hasattr(target, "weight"):
                    skipped += 1
                    continue
                target.weight.data += chunk.to(target.weight.device)
                injected += 1

        logger.info(
            f"Manually injected LoRA into {injected} modules "
            f"({skipped} skipped/not found)"
        )

    @staticmethod
    def _resolve_module(root, path: str):
        """Resolve a dotted module path like 'transformer_blocks.0.attn.to_q'."""
        target = root
        try:
            for p in path.split("."):
                if p.isdigit():
                    target = target[int(p)]
                else:
                    target = getattr(target, p)
            return target
        except (AttributeError, IndexError, KeyError):
            return None

    @staticmethod
    def _compute_delta(matrices: Dict, weight) -> "torch.Tensor":
        """Compute LoRA delta (B @ A) on GPU, move to weight's device."""
        import torch
        compute_device = "cuda" if torch.cuda.is_available() else weight.device
        A = matrices["lora_A"].to(weight.dtype).to(compute_device)
        B = matrices["lora_B"].to(weight.dtype).to(compute_device)
        return (B @ A).to(weight.device)

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

        # Build prompt and conditioning images
        # ml-sharp LoRA expects exactly 2 images:
        #   Image 1: Reference scene (forward warp = original perspective)
        #   Image 2: Scene to repair (splat render with gaps)
        # See: huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair
        if self._backend_name == "klein-sharp-lora":
            transform = compute_relative_transform(inputs.ref_c2w, inputs.novel_c2w)
            prompt = build_repair_prompt(transform)
        else:
            prompt = (
                "Referring to the scene in image 1, restore the perspective of "
                "the scene in image 2. Repair the perspective and missing areas."
            )

        images = [
            self._to_pil(inputs.forward_warped_rgb, size=(1024, 1024)),
            self._to_pil(inputs.broken_splat_render, size=(1024, 1024)),
        ]
        logger.info(f"Prompt: {prompt[:80]}...")

        result = self._pipeline(
            prompt=prompt,
            image=images,
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
