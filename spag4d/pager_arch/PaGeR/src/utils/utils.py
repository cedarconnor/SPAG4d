"""Logging / config helpers shared by inference.py, app.py, and the evaluation scripts."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from scipy.ndimage import median_filter


def args_to_omegaconf(args, base_cfg):
    """Recursively overlay non-None argparse values onto base_cfg, matched by leaf-key name."""
    cfg = OmegaConf.create(base_cfg)

    def _override(container, key):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                container[key] = value

    def _walk(container):
        if not isinstance(container, DictConfig):
            return
        for key in container.keys():
            node = container[key]
            if isinstance(node, DictConfig):
                _walk(node)
            else:
                _override(container, key)

    _walk(cfg)
    return cfg


def convert_paths_to_pathlib(cfg):
    """Cast leaf keys whose name contains 'path' to pathlib.Path."""
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = convert_paths_to_pathlib(value)
        elif "path" in key.lower():
            cfg[key] = Path(value) if value is not None else None
    return cfg


def prepare_image_for_logging(image: np.ndarray, apply_median: bool = False) -> np.ndarray:
    """Per-sample min/max stretch to uint8, optionally median-filtered."""
    if apply_median:
        image = median_filter(image.astype(np.float32), size=3)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return (image * 255).astype("uint8")


def prepare_depth_for_logging(
    pager,
    depth,
    sky_mask,
    orig_size: Tuple[int, int],
    cmap,
    log_scale=None,
    sky_mask_threshold: float = 0.3,
    sky_mask_softness: float = 0.15,
    sky_mask_open_kernel: int = 0,
):
    """Return (raw_depth float32 ERP, viz_depth uint8 Spectral-coloured ERP)."""
    raw_image, log_image = pager.process_depth_output(
        depth,
        orig_size=orig_size,
        sky_mask=sky_mask,
        log_scale=log_scale,
        sky_mask_threshold=sky_mask_threshold,
        sky_mask_softness=sky_mask_softness,
        sky_mask_open_kernel=sky_mask_open_kernel,
    )
    raw_image = raw_image.detach().cpu().numpy()
    log_image = log_image.detach().cpu().numpy()
    viz = prepare_image_for_logging(log_image, apply_median=True)
    viz = cmap(viz / 255.0)
    viz = (viz[..., :3] * 255)[0].astype(np.uint8)
    return raw_image, np.transpose(viz, (2, 0, 1))


def prepare_normals_for_logging(
    pager,
    normals,
    sky_mask,
    orig_size: Tuple[int, int],
    cmap=None,
    sky_mask_threshold: float = 0.3,
    sky_mask_softness: float = 0.15,
    sky_mask_open_kernel: int = 0,
):
    """Return (raw_normals (3, H, W) float32 in [-1, 1], viz_normals uint8 RGB)."""
    raw_normals = pager.process_normals_output(
        normals,
        orig_size=orig_size,
        sky_mask=sky_mask,
        sky_mask_threshold=sky_mask_threshold,
        sky_mask_softness=sky_mask_softness,
        sky_mask_open_kernel=sky_mask_open_kernel,
    ).detach().cpu().numpy()
    viz = prepare_image_for_logging(raw_normals)
    return raw_normals, viz
