# spag4d/pager_model.py
"""PaGeR depth backend wrapper. Same role/contract as DA360Model / DAPModel.

Wraps the vendored unified prs-eth/PaGeR (DA3 cubemap multi-view) model.
``predict()`` returns ``(depth, sky_mask)`` to fit the existing ``(depth, mask)``
slot used by core.py; surface normals + native resolution + convention are
stashed on the instance for core.py to pull.

Pipeline inside predict():
    ERP image (H,W,3)
      → ImageNet-normalize → erp_to_cubemap (6,3,504,504)
      → Pager.forward(skip_heads=...) → raw per-face heads
      → process_depth_output(orig_size=(H,W))  → radial/Euclidean ERP depth
      → process_normals_output(orig_size=(H,W)) → world-frame unit normals (H,W,3)
      → sigmoid(sky logits) stitched to ERP → binary sky mask (H,W)

Scale-invariant depth is the default (both scale heads skipped, log_scale=None).
``metric=True`` routes one scale head via a CLIP indoor/outdoor classifier.

Model weights are CC BY-NC 4.0 — non-commercial / evaluation use only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .pager_arch import build_pager, _ensure_on_path

PAGER_REPO = "prs-eth/PaGeR"
PAGER_CACHE_DIR = Path.home() / ".cache" / "spag4d" / "pager"

# ImageNet stats — PaGeR's dataloader normalizes the ERP before cubemap projection.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Both scale heads share the "scale" output key; exactly one may run per forward.
_SCALE_HEADS = {"indoor": "scale_indoor", "outdoor": "scale_outdoor"}


def _load_cfg():
    """Load the checkpoint's model config (carries modalities, face_size, log_depth)."""
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    path = hf_hub_download(PAGER_REPO, "config.yaml", cache_dir=str(PAGER_CACHE_DIR))
    return OmegaConf.load(path)


def _erp_to_cubemap(chw: torch.Tensor, face_w: int, fov: float) -> torch.Tensor:
    """chw: (3,H,W) ImageNet-normalized ERP. Returns (6,3,face,face), order [F,R,B,L,U,D]."""
    _ensure_on_path()
    from src.utils.geometry_utils import erp_to_cubemap
    return erp_to_cubemap(chw, face_w=face_w, fov=fov)


def _cubemap_to_erp(cube: torch.Tensor, H: int, W: int, fov: float) -> torch.Tensor:
    """Stitch a (6,C,face,face) cube to a (C,H,W) ERP map (used for the sky mask)."""
    _ensure_on_path()
    from src.utils.geometry_utils import cubemap_to_erp
    return cubemap_to_erp(cube, erp_h=H, erp_w=W, fov=fov)


def _get_classifier(device):
    _ensure_on_path()
    from src.utils.scene_classifier import get_classifier
    return get_classifier(device=device)


class PaGeRModel:
    """PaGeR geometry backend. Mirrors DA360Model/DAPModel; predict()→(depth, sky_mask)."""

    def __init__(self, pager, cfg, device: torch.device, metric: bool = False):
        self._pager = pager
        self._cfg = cfg
        self.device = device
        self.metric = metric
        self.depth_convention = "radial"
        self.last_normals: torch.Tensor | None = None
        # Detail is capped at face_size² per cube face regardless of ERP output size.
        # Record the effective native ERP (≈4 equatorial faces wide, 2 tall) for honest logging.
        face = int(getattr(cfg, "face_size", 504))
        self.native_resolution: tuple[int, int] = (2 * face, 4 * face)
        self._classifier = None

    @classmethod
    def load(cls, device: torch.device = torch.device("cuda"),
             metric: bool = False) -> "PaGeRModel":
        """Lazy-load the unified prs-eth/PaGeR checkpoint from the HF cache."""
        cfg = _load_cfg()
        pager = build_pager(PAGER_REPO, cfg=cfg, device=device)
        pager.get_intrinsics_extrinsics(
            image_size=cfg.face_size, fov=getattr(cfg, "cube_fov", 90.0))
        # Widen the depth clamp so outdoor scenes aren't truncated at the 75 m indoor default.
        pager.set_depth_range(1e-2, 200.0)
        pager.model.to(device, dtype=getattr(pager, "weight_dtype", torch.float32))
        pager.model.eval()
        return cls(pager, cfg, device, metric=metric)

    def _normalize(self, chw: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(_IMAGENET_MEAN, device=chw.device).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=chw.device).view(3, 1, 1)
        return (chw - mean) / std

    def _skip_heads(self, cube: torch.Tensor):
        """Return (skip_heads, use_log_scale). SI depth skips both scale heads."""
        if not self.metric:
            return {"scale_indoor", "scale_outdoor"}, False
        # Metric: route one scale head via CLIP indoor/outdoor on de-normalized faces.
        if self._classifier is None:
            self._classifier = _get_classifier(self.device)
        mean = torch.tensor(_IMAGENET_MEAN, device=cube.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=cube.device).view(1, 3, 1, 1)
        cube01 = cube[0] * std + mean  # (6,3,h,w) in ~[0,1]
        label, _ = self._classifier.classify(cube01)
        keep = _SCALE_HEADS[label]
        return {h for h in _SCALE_HEADS.values() if h != keep}, True

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """image: (H,W,3) uint8/float on device. Returns (depth, sky_mask).

        depth: (H,W) float32 radial/Euclidean ERP depth at working resolution.
        sky_mask: (H,W) bool (sigmoid(sky logits) > 0.5).
        Side-effects: self.last_normals (H,W,3) world-frame unit; native_resolution.
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        H, W = int(image.shape[0]), int(image.shape[1])

        chw = image.permute(2, 0, 1).to(self.device)          # (3,H,W) in [0,1]
        chw = self._normalize(chw)
        cube = _erp_to_cubemap(chw, int(self._cfg.face_size),
                               float(getattr(self._cfg, "cube_fov", 90.0)))
        cube = cube.unsqueeze(0).to(self.device)              # (1,6,3,face,face)

        skip, use_scale = self._skip_heads(cube)
        pred = self._pager(cube, dtype=torch.float16, skip_heads=skip)

        log_scale = pred.get("scale") if use_scale else None
        sky_logits = pred["sky"][0]  # (6,1,face,face)

        # Depth: sky-filled to MAX_DEPTH so the existing depth-threshold sky drop works.
        depth_erp, _ = self._pager.process_depth_output(
            pred["depth"][0], (H, W), sky_mask=sky_logits, log_scale=log_scale)
        depth = torch.as_tensor(depth_erp).float()
        if depth.dim() == 3:
            depth = depth[0]                                  # (1,H,W) -> (H,W)

        # Normals: world-frame unit map (3,H,W) -> (H,W,3), renormalized for safety.
        normals_erp = self._pager.process_normals_output(pred["normals"][0], (H, W))
        n = torch.as_tensor(normals_erp).float()
        if n.dim() == 3 and n.shape[0] == 3:
            n = n.permute(1, 2, 0)                            # (H,W,3)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.last_normals = n

        # Binary sky mask at working resolution: stitch sigmoid(logits) to ERP.
        sky_prob_cube = torch.sigmoid(sky_logits.float())     # (6,1,face,face)
        sky_prob_erp = _cubemap_to_erp(
            sky_prob_cube, H, W, float(getattr(self._cfg, "cube_fov", 90.0)))
        sky_prob_erp = torch.as_tensor(sky_prob_erp).float()
        if sky_prob_erp.dim() == 3:
            sky_prob_erp = sky_prob_erp[0]                    # (1,H,W) -> (H,W)
        sky_mask = (sky_prob_erp > 0.5).bool()

        return depth, sky_mask
