"""Diagnostic bundle rendering for visual comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def save_diagnostic_bundle(
    splat_rgb: np.ndarray,
    warp_rgb: np.ndarray,
    pano_rgb: Optional[np.ndarray],
    region_map: Optional[np.ndarray] = None,
    output_path: str | Path = "diagnostic.png",
    labels: Optional[list[str]] = None,
) -> None:
    """
    Save a side-by-side comparison image.

    Args:
        splat_rgb: [H, W, 3] float32 — gsplat render
        warp_rgb: [H, W, 3] float32 — forward warp
        pano_rgb: [H, W, 3] float32 — panoramic extraction (or None)
        region_map: [H, W] int — region classification overlay (optional)
        output_path: Where to save the image.
        labels: Optional labels for each panel.
    """
    panels = [splat_rgb, warp_rgb]
    if pano_rgb is not None:
        panels.append(pano_rgb)

    if region_map is not None:
        # Color-code regions
        H, W = region_map.shape
        region_vis = np.zeros((H, W, 3), dtype=np.float32)
        # TRUSTED = green, TYPE_A = yellow, TYPE_B = orange, TYPE_C = red
        colors = {
            0: [0.0, 0.8, 0.0],   # TRUSTED
            1: [0.9, 0.9, 0.0],   # TYPE_A
            2: [1.0, 0.5, 0.0],   # TYPE_B
            3: [1.0, 0.0, 0.0],   # TYPE_C
        }
        for val, color in colors.items():
            mask = region_map == val
            region_vis[mask] = color
        panels.append(region_vis)

    # Resize all panels to match the smallest height
    min_h = min(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != min_h:
            scale = min_h / p.shape[0]
            new_w = int(p.shape[1] * scale)
            img = Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8))
            img = img.resize((new_w, min_h), Image.LANCZOS)
            resized.append(np.array(img).astype(np.float32) / 255.0)
        else:
            resized.append(p)

    # Concatenate horizontally with 2px separator
    gap = np.ones((min_h, 2, 3), dtype=np.float32)
    parts = []
    for i, panel in enumerate(resized):
        if i > 0:
            parts.append(gap)
        parts.append(panel)

    combined = np.concatenate(parts, axis=1)
    img = Image.fromarray((np.clip(combined, 0, 1) * 255).astype(np.uint8))
    img.save(str(output_path))
