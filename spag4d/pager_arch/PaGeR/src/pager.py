# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The PaGeR Authors.
"""PaGeR inference wrapper around the Depth Anything 3 backbone with depth / normals / sky / scale heads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

from src.depth_anything_3.api import DepthAnything3
from src.utils.geometry_utils import (
    cubemap_to_erp,
    get_cubemap_intrinsics_extrinsics,
    unit_normals,
    z_depth_to_euclidean,
)


def _tile_to_shape(src: torch.Tensor, tgt_shape: Sequence[int]) -> Optional[torch.Tensor]:
    """Tile ``src`` to ``tgt_shape``, dividing by repetitions on dim>=1 to preserve magnitude.

    Returns ``None`` if a target dim isn't an integer multiple of the source dim.
    """
    w = src.clone()
    for dim in range(len(tgt_shape)):
        if w.shape[dim] == tgt_shape[dim]:
            continue
        if tgt_shape[dim] % w.shape[dim] != 0:
            return None
        reps = tgt_shape[dim] // w.shape[dim]
        rep_spec = [1] * w.ndim
        rep_spec[dim] = reps
        w = w.repeat(*rep_spec)
        if w.ndim > 1 and dim >= 1:
            w = w / reps
    return w


class Pager(nn.Module):
    """Inference wrapper around the PaGeR multi-head model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg,
        device: torch.device = torch.device("cpu"),
        weight_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight_dtype = weight_dtype
        self.device = device
        self.modalities = list(cfg.modalities)
        # Released checkpoints all train z-depth in log space.
        self.z_depth = True
        self.log_depth = bool(cfg.log_depth)
        self.cube_fov = float(getattr(cfg, "cube_fov", 90.0))
        self.set_depth_range(1e-2, 75.0)

        self._build_model(checkpoint_path, cfg)

    def _build_model(self, checkpoint_path: str | Path, model_cfg) -> None:
        # Supports the released multi-head format and the base depth-only DA3 checkpoint
        # (split + warm-start non-depth heads from tiled depth weights).
        checkpoint_path = Path(checkpoint_path)
        local_safetensors = checkpoint_path / "model.safetensors"

        if local_safetensors.exists():
            state_dict = load_file(local_safetensors)
            ckpt_heads = self._detect_ckpt_heads(state_dict)
            is_multihead_fmt = any(
                k.startswith("model.head.heads.") for k in state_dict
            )

            if not is_multihead_fmt:
                # Depth-only checkpoint: DA3 from_pretrained broadcasts depth weights
                # into every head slot; the split below categorises them.
                del state_dict
                self.model = DepthAnything3.from_pretrained(
                    checkpoint_path, model_cfg=model_cfg
                )
                self._split_into_independent_heads(
                    model_cfg, pretrained_sources=ckpt_heads
                )
                return

            pretrained = str(getattr(model_cfg, "pretrained_model_name", "da3-giant"))
            self.model = DepthAnything3(model_name=pretrained, model_cfg=model_cfg)
            if "depth" in ckpt_heads:
                self._preload_depth_into_old_head(state_dict)
            self._split_into_independent_heads(
                model_cfg, pretrained_sources=ckpt_heads
            )
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self._report_load_diff(missing, unexpected)
            return

        # Hub fallback: dispatch through the same loader so cfg.pretrained_model_name
        # (e.g. da3-giant) wins over the mixin's hard-coded da3-large default.
        try:
            sf_path = hf_hub_download(
                repo_id=str(checkpoint_path), filename="model.safetensors"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve PaGeR checkpoint '{checkpoint_path}': "
                f"no local model.safetensors and HF download failed ({e})."
            ) from e

        state_dict = load_file(sf_path)
        ckpt_heads = self._detect_ckpt_heads(state_dict)
        is_multihead_fmt = any(
            k.startswith("model.head.heads.") for k in state_dict
        )

        if not is_multihead_fmt:
            del state_dict
            self.model = DepthAnything3.from_pretrained(
                checkpoint_path, model_cfg=model_cfg
            )
            self._split_into_independent_heads(
                model_cfg, pretrained_sources=ckpt_heads
            )
            return

        pretrained = str(getattr(model_cfg, "pretrained_model_name", "da3-giant"))
        self.model = DepthAnything3(model_name=pretrained, model_cfg=model_cfg)
        if "depth" in ckpt_heads:
            self._preload_depth_into_old_head(state_dict)
        self._split_into_independent_heads(
            model_cfg, pretrained_sources=ckpt_heads
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self._report_load_diff(missing, unexpected)

    # cam_dec.* trains alongside the model but isn't loaded at inference; silence its keys.
    _EXPECTED_UNEXPECTED_PREFIXES = ("model.cam_dec.",)

    @classmethod
    def _report_load_diff(cls, missing, unexpected) -> None:
        unexpected = [
            k for k in unexpected
            if not k.startswith(cls._EXPECTED_UNEXPECTED_PREFIXES)
        ]
        if missing:
            print(
                f"[Pager] Warning: {len(missing)} missing keys when loading checkpoint "
                f"(first 5: {missing[:5]})"
            )
        if unexpected:
            print(
                f"[Pager] Warning: {len(unexpected)} unexpected keys when loading checkpoint "
                f"(first 5: {unexpected[:5]})"
            )

    @staticmethod
    def _detect_ckpt_heads(state_dict) -> set:
        heads: set = set()
        known = ("depth", "normals", "sky", "scale", "scale_indoor", "scale_outdoor")
        for k in state_dict:
            if k.startswith("model.head.heads."):
                heads.add(k[len("model.head.heads."):].split(".", 1)[0])
                continue
            if ".scratch." in k:
                for h in known:
                    if f".scratch.{h}_output_conv2." in k:
                        heads.add(h)
                        break
                else:
                    if ".scratch.output_conv2." in k:
                        heads.add("depth")
        return heads & set(known)

    def _preload_depth_into_old_head(self, state_dict) -> None:
        # Seed the pre-split DualDPT with depth weights so the split's transfer step has a real source.
        src_prefix = "model.head.heads.depth."
        tgt_prefix = "model.head."
        remapped = {
            tgt_prefix + k[len(src_prefix):]: v
            for k, v in state_dict.items() if k.startswith(src_prefix)
        }
        self.model.load_state_dict(remapped, strict=False)

    def _split_into_independent_heads(
        self, model_cfg, pretrained_sources: Optional[Iterable[str]] = None
    ) -> None:
        """Split the shared DualDPT into one head per modality (warm-starting from depth where needed)."""
        from src.depth_anything_3.model.dualdpt import DualDPT, MultiDualDPT
        from src.depth_anything_3.model.scale_head_dpt import ScaleHeadDPT
        from src.depth_anything_3.model.sky_head import SkyBranchHead

        old_head = self.model.model.head

        pretrained_sources = (
            set(old_head.head_modules.keys())
            if pretrained_sources is None
            else set(pretrained_sources)
        )

        head_cfg = OmegaConf.to_container(self.model.config.head, resolve=True)
        head_cfg.pop("__object__", None)
        head_cfg.pop("head_names", None)
        head_cfg.pop("global_PE", None)

        token_dim = head_cfg.get("dim_in", 3072)

        new_heads: dict[str, nn.Module] = {}

        for modality in model_cfg.modalities:
            if modality == "sky":
                new_heads[modality] = SkyBranchHead(token_dim=token_dim)
                continue

            if modality in ("scale", "scale_indoor", "scale_outdoor"):
                head_module = ScaleHeadDPT(
                    token_dim=token_dim,
                    out_channels=head_cfg.get("out_channels", (256, 512, 1024, 1024)),
                    features=head_cfg.get("features", 256),
                    patch_size=head_cfg.get("patch_size", 14),
                    downsample_factor=int(getattr(
                        model_cfg, "scale_head_downsample_factor", 4)),
                )
                if "depth" in old_head.head_modules:
                    head_module.init_from_depth(old_head)
                new_heads[modality] = head_module
                continue

            new_dpt = DualDPT(head_names=[modality], **head_cfg)

            if modality in pretrained_sources and modality in old_head.head_modules:
                new_dpt.norm.load_state_dict(old_head.norm.state_dict())
                new_dpt.projects.load_state_dict(old_head.projects.state_dict())
                new_dpt.resize_layers.load_state_dict(old_head.resize_layers.state_dict())
                new_dpt.head_modules[modality].load_state_dict(
                    old_head.head_modules[modality].state_dict()
                )
            elif modality in old_head.head_modules:
                # Slot already holds depth-tiled weights from the DA3 scratch-broadcast path.
                new_dpt.norm.load_state_dict(old_head.norm.state_dict())
                new_dpt.projects.load_state_dict(old_head.projects.state_dict())
                new_dpt.resize_layers.load_state_dict(old_head.resize_layers.state_dict())
                new_dpt.head_modules[modality].load_state_dict(
                    old_head.head_modules[modality].state_dict()
                )
            elif "depth" in old_head.head_modules:
                self._init_head_from_depth(new_dpt, old_head, modality)

            new_heads[modality] = new_dpt

        multi_head = MultiDualDPT(new_heads)

        if model_cfg.valid_conv_padding:
            from src.depth_anything_3.model.utils.valid_conv_padding import (
                set_valid_pad_conv,
            )
            set_valid_pad_conv(multi_head, fov_deg=self.cube_fov)

        self.model.model.head = multi_head

    @staticmethod
    def _init_head_from_depth(new_dpt, old_head, modality: str) -> None:
        """Warm-start a non-depth head by tiling the trained depth weights into its slot."""
        depth_mod = old_head.head_modules["depth"]
        new_dpt.norm.load_state_dict(old_head.norm.state_dict())
        new_dpt.projects.load_state_dict(old_head.projects.state_dict())
        new_dpt.resize_layers.load_state_dict(old_head.resize_layers.state_dict())

        src_sd = depth_mod.state_dict()
        tgt_sd = new_dpt.head_modules[modality].state_dict()
        for key in tgt_sd:
            if key not in src_sd:
                continue
            src_w = src_sd[key]
            tgt_shape = tgt_sd[key].shape
            # Depth's final conv can be 2-channel (depth + confidence); only the depth channel transfers.
            if key in ("output_conv2.2.weight", "output_conv2.2.bias") and src_w.shape[0] > 1:
                src_w = src_w[:1]
            if src_w.shape == tgt_shape:
                tgt_sd[key] = src_w.clone()
            elif key.startswith("output_conv2."):
                tiled = _tile_to_shape(src_w, tgt_shape)
                if tiled is not None:
                    tgt_sd[key] = tiled
        new_dpt.head_modules[modality].load_state_dict(tgt_sd)

    def get_intrinsics_extrinsics(self, image_size: int = 504, fov: float = 90.0) -> None:
        """Build and cache the six cubemap cameras. Must be called before the first forward pass."""
        extrinsics, intrinsics = get_cubemap_intrinsics_extrinsics(image_size, fov=fov)
        self.extrinsics = extrinsics.to(self.device)
        self.intrinsics = intrinsics.to(self.device)

    def set_depth_range(self, min_depth: float, max_depth: float) -> None:
        # Outdoor datasets use a larger ceiling than the 75 m indoor default.
        self.MIN_DEPTH = float(min_depth)
        self.MAX_DEPTH = float(max_depth)
        self.LOG_MIN_DEPTH = math.log(self.MIN_DEPTH)
        self.LOG_MAX_DEPTH = math.log(self.MAX_DEPTH)

    def forward(
        self,
        rgb_cubemap: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        skip_heads: Optional[Iterable[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass on a (B, 6, 3, H, W) ImageNet-normalised cubemap [F, R, B, L, U, D].

        Returns raw outputs (log-z-depth, sky logits, ...); use process_depth_output /
        process_normals_output for ERP-stitched, sky-masked predictions.
        """
        ext = self.extrinsics
        intr = self.intrinsics
        with torch.autocast(device_type=self.device.type, dtype=dtype):
            return self.model(
                image=rgb_cubemap,
                extrinsics=ext,
                intrinsics=intr,
                face_ids=None,
                skip_heads=skip_heads,
            )

    def _sky_blend_alpha(
        self,
        sky_logits: torch.Tensor,
        threshold: float,
        softness: float,
        open_kernel: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sky logits → soft blending alpha (sigmoid, optional morphological opening, smoothstep)."""
        sky_prob = torch.sigmoid(sky_logits)
        speck_mask = None
        if open_kernel and open_kernel > 1:
            binary_thresh = max(0.0, float(threshold) - float(softness))
            binary = (sky_prob > binary_thresh).to(sky_prob.dtype)
            orig_shape = binary.shape
            x = binary.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
            pad = int(open_kernel) // 2
            xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), value=1.0)
            x = -torch.nn.functional.max_pool2d(-xp, kernel_size=int(open_kernel), stride=1)
            xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), value=0.0)
            keep = torch.nn.functional.max_pool2d(xp, kernel_size=int(open_kernel), stride=1).reshape(orig_shape)
            speck_mask = (binary > 0) & (keep <= 0)
            sky_prob = sky_prob * keep
        if softness > 0:
            t = ((sky_prob - (threshold - softness)) / (2.0 * softness)).clamp(0.0, 1.0)
            alpha = t * t * (3.0 - 2.0 * t)
        else:
            alpha = (sky_prob > threshold).to(sky_prob.dtype)
        occupancy = (sky_prob > threshold).sum() / sky_prob.numel()
        return alpha, occupancy, speck_mask

    def process_depth_output(
        self,
        pred_cubemap: torch.Tensor,
        orig_size: tuple[int, int],
        sky_mask: Optional[torch.Tensor] = None,
        log_scale: Optional[torch.Tensor] = None,
        sky_mask_threshold: float = 0.3,
        sky_mask_softness: float = 0.15,
        sky_mask_open_kernel: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw depth head output → metric ERP depth.

        clamp/unlog → scale → z→euclidean → sky-fill → cubemap-to-ERP. Returns (linear, log) at orig_size.
        """
        if self.log_depth:
            pred_cubemap = torch.clamp(pred_cubemap, self.LOG_MIN_DEPTH, self.LOG_MAX_DEPTH)
            pred_cubemap = torch.exp(pred_cubemap)
        else:
            pred_cubemap = torch.clamp(pred_cubemap, self.MIN_DEPTH, self.MAX_DEPTH)

        if log_scale is not None:
            if log_scale.shape[-1] == 2:
                # Uncertainty-aware scale head emits (mu, log_sigma); only mu drives rescaling.
                log_scale = log_scale[..., :1]
            scale_factor = torch.exp(log_scale).view(-1, *([1] * (pred_cubemap.ndim - 1)))
            pred_cubemap = pred_cubemap * scale_factor

        if self.z_depth:
            pred_cubemap = z_depth_to_euclidean(pred_cubemap, self.intrinsics)

        if sky_mask is not None:
            alpha, occupancy, speck_mask = self._sky_blend_alpha(
                sky_mask, sky_mask_threshold, sky_mask_softness, sky_mask_open_kernel
            )
            # Min-pool object depth into killed specks before the blend so alpha→0
            # doesn't expose wrong-depth blobs under sky-confused pixels.
            if speck_mask is not None and speck_mask.any():
                k = int(sky_mask_open_kernel)
                if k % 2 == 0:
                    k += 1
                pad = k // 2
                orig_shape = pred_cubemap.shape
                x = pred_cubemap.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
                xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
                x_min = (-torch.nn.functional.max_pool2d(-xp, kernel_size=k, stride=1)).reshape(orig_shape)
                sm = speck_mask.to(pred_cubemap.dtype)
                pred_cubemap = pred_cubemap * (1.0 - sm) + x_min * sm
            if occupancy > 0.01:
                # Sky sits at post-scale MAX_DEPTH, so log_max needs the matching log_scale offset
                # — otherwise outdoor sky stays pinned at the indoor ceiling.
                log_max = math.log(self.MAX_DEPTH)
                if log_scale is not None:
                    log_max = log_max + log_scale.view(
                        -1, *([1] * (pred_cubemap.ndim - 1))
                    ).to(pred_cubemap.dtype)
                pred_log = torch.log(pred_cubemap.clamp_min(1e-6))
                alpha = alpha.to(pred_log.dtype)
                pred_cubemap = torch.exp((1.0 - alpha) * pred_log + alpha * log_max)

        # grid_sample in the ERP stitch is fp16-unstable along face seams; promote to fp32.
        pred_erp = cubemap_to_erp(pred_cubemap.float(), *orig_size, fov=self.cube_fov)
        return pred_erp, torch.log(pred_erp)

    def process_normals_output(
        self,
        pred_cubemap: torch.Tensor,
        orig_size: tuple[int, int],
        sky_mask: Optional[torch.Tensor] = None,
        sky_mask_threshold: float = 0.3,
        sky_mask_softness: float = 0.15,
        sky_mask_open_kernel: int = 0,
    ) -> torch.Tensor:
        """Raw normals head output → unit-length ERP map."""
        if sky_mask is not None:
            alpha, occupancy, _ = self._sky_blend_alpha(
                sky_mask, sky_mask_threshold, sky_mask_softness, sky_mask_open_kernel
            )
            if occupancy > 0.01:
                pred_cubemap = pred_cubemap * (1.0 - alpha).to(pred_cubemap.dtype)
        pred_erp = cubemap_to_erp(pred_cubemap.float(), *orig_size, fov=self.cube_fov)
        return unit_normals(pred_erp)
