"""Convention-aware gsplat renderer wrapper."""

from __future__ import annotations

# Bootstrap MSVC environment before gsplat import (Windows JIT compilation)
import spag4d_refine.msvc_env  # noqa: F401

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..gaussian.cloud import GaussianCloud
from ..camera.pinhole import PinholeCamera


@dataclass
class RenderResult:
    """Result of rendering a Gaussian cloud."""
    rgb: np.ndarray      # [H, W, 3] float32 [0, 1]
    depth: np.ndarray    # [H, W] float32
    alpha: np.ndarray    # [H, W] float32 [0, 1]


class GsplatRenderer:
    """
    Render a GaussianCloud using gsplat rasterization.

    Handles convention conversion:
    - Quaternions: XYZW (internal) → WXYZ (gsplat)
    - Scales: linear (internal) — gsplat expects linear
    - Opacities: [0,1] (internal) — gsplat expects [0,1]
    """

    def __init__(self, cloud: GaussianCloud, device: str = "cuda"):
        self.device = torch.device(device)
        gsplat_data = cloud.to_gsplat()
        self.means = gsplat_data["means"].to(self.device)
        self.scales = gsplat_data["scales"].to(self.device)
        self.quats = gsplat_data["quats"].to(self.device)  # WXYZ
        self.colors = gsplat_data["colors"].to(self.device)
        self.opacities = gsplat_data["opacities"].to(self.device)

    def render(self, camera: PinholeCamera) -> RenderResult:
        """Render the Gaussian cloud from the given camera."""
        from gsplat import rasterization

        H, W = camera.height, camera.width

        # Build viewmat [1, 4, 4] and K [1, 3, 3]
        viewmat = torch.from_numpy(camera.w2c).float().unsqueeze(0).to(self.device)
        K = torch.from_numpy(camera.K).float().unsqueeze(0).to(self.device)

        # SH coefficients: degree 0 → just colors as [N, 1, 3]
        sh0 = self.colors.unsqueeze(1)

        renders, alphas, info = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=sh0,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=0,
            render_mode="RGB+ED",
        )

        # renders: [1, H, W, 4] (RGB + expected depth)
        # alphas: [1, H, W, 1]
        rgb = renders[0, :, :, :3].clamp(0, 1).cpu().numpy()
        depth = renders[0, :, :, 3].cpu().numpy()
        alpha = alphas[0, :, :, 0].cpu().numpy()

        return RenderResult(rgb=rgb, depth=depth, alpha=alpha)
