"""GaussianCloud: unified Gaussian splat container with provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from plyfile import PlyData, PlyElement

from .conventions import ConventionFlags, detect_conventions
from .provenance import GaussianSource

# Spherical harmonics DC normalization constant
SH_C0 = 0.28209479177387814


def _linearRGB_to_sRGB(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB OETF."""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.clip(linear, 0.0031308, None) ** (1.0 / 2.4) - 0.055,
    )


class GaussianCloud:
    """
    Gaussian splat container with provenance tracking.

    Internal conventions (matching SPAG-4D):
    - Quaternions: XYZW order
    - Scales: linear (not log)
    - Opacities: [0, 1] (not logit)
    - Colors: sRGB [0, 1]
    - Coordinate system: Y-up
    """

    def __init__(
        self,
        means: np.ndarray,       # [N, 3]
        scales: np.ndarray,      # [N, 3] linear
        quats: np.ndarray,       # [N, 4] XYZW
        colors: np.ndarray,      # [N, 3] sRGB [0, 1]
        opacities: np.ndarray,   # [N, 1] or [N] in [0, 1]
        provenance: Optional[np.ndarray] = None,  # [N] GaussianSource values
    ):
        self.means = np.asarray(means, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.quats = np.asarray(quats, dtype=np.float32)
        self.colors = np.asarray(colors, dtype=np.float32)

        opacities = np.asarray(opacities, dtype=np.float32)
        if opacities.ndim == 1:
            opacities = opacities[:, np.newaxis]
        self.opacities = opacities

        if provenance is None:
            self.provenance = np.full(len(self.means), GaussianSource.ORIGINAL, dtype=np.int32)
        else:
            self.provenance = np.asarray(provenance, dtype=np.int32)

        # Normalize quaternions
        qnorm = np.linalg.norm(self.quats, axis=-1, keepdims=True)
        qnorm = np.clip(qnorm, 1e-8, None)
        self.quats = self.quats / qnorm

    def __len__(self) -> int:
        return self.means.shape[0]

    @classmethod
    def from_ply(cls, path: str | Path) -> GaussianCloud:
        """
        Load a standard 3DGS PLY file.

        Auto-detects conventions and converts to internal format.
        """
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data

        conventions = detect_conventions(vertex)

        # Positions
        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

        # Scales
        raw_scales = np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
        )
        if conventions.scale_encoding == "log":
            scales = np.exp(raw_scales)
        else:
            scales = raw_scales

        # Colors from SH DC
        sh_dc = np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
        )
        colors = np.clip(sh_dc * SH_C0 + 0.5, 0.0, 1.0)

        # Opacities
        raw_opacity = vertex["opacity"]
        if conventions.opacity_encoding == "logit":
            opacities = 1.0 / (1.0 + np.exp(-raw_opacity))
        else:
            opacities = raw_opacity

        # Quaternions → internal XYZW
        if conventions.quat_order == "WXYZ":
            quats = np.stack(
                [vertex["rot_1"], vertex["rot_2"], vertex["rot_3"], vertex["rot_0"]],
                axis=-1,
            )
        else:
            quats = np.stack(
                [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
                axis=-1,
            )

        return cls(
            means=means,
            scales=scales,
            quats=quats,
            colors=colors,
            opacities=opacities,
        )

    def to_ply(self, path: str | Path, colors_linear: bool = False) -> None:
        """
        Save to standard 3DGS PLY format.

        PLY conventions: WXYZ quaternions, log scales, logit opacities, SH0 colors.
        """
        N = len(self)
        if N == 0:
            raise ValueError("No Gaussians to save")

        # Log scales
        log_scales = np.log(np.clip(self.scales, 1e-7, None))

        # Colors → SH DC
        colors_clamped = np.clip(self.colors, 0.0, 1.0)
        if colors_linear:
            colors_srgb = _linearRGB_to_sRGB(colors_clamped)
        else:
            colors_srgb = colors_clamped
        sh_dc = (colors_srgb - 0.5) / SH_C0

        # Logit opacities
        op = np.clip(self.opacities.squeeze(), 1e-6, 1 - 1e-6)
        opacity_logit = np.log(op / (1 - op))

        # Build structured array
        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]

        data = np.zeros(N, dtype=dtype_list)
        data["x"], data["y"], data["z"] = self.means.T
        data["nx"] = data["ny"] = data["nz"] = 0
        data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = sh_dc.T
        data["opacity"] = opacity_logit
        data["scale_0"], data["scale_1"], data["scale_2"] = log_scales.T

        # XYZW → WXYZ for PLY
        data["rot_0"] = self.quats[:, 3]  # W
        data["rot_1"] = self.quats[:, 0]  # X
        data["rot_2"] = self.quats[:, 1]  # Y
        data["rot_3"] = self.quats[:, 2]  # Z

        el = PlyElement.describe(data, "vertex")
        PlyData([el], text=False).write(str(path))

    # Provenance → color mapping for heatmap visualization
    _PROVENANCE_COLORS = {
        int(GaussianSource.ORIGINAL): np.array([0.2, 0.3, 0.9], dtype=np.float32),
        int(GaussianSource.SEEDED):   np.array([0.9, 0.85, 0.1], dtype=np.float32),
        int(GaussianSource.PROMOTED): np.array([0.1, 0.85, 0.2], dtype=np.float32),
        int(GaussianSource.PRUNED):   np.array([0.9, 0.1, 0.1], dtype=np.float32),
    }

    def to_heatmap_ply(self, path: str | Path) -> None:
        """Save PLY with colors encoding provenance (blue=original, green=promoted, yellow=seeded)."""
        colors = np.zeros((len(self), 3), dtype=np.float32)
        for source_int, color in self._PROVENANCE_COLORS.items():
            mask = self.provenance == source_int
            colors[mask] = color

        heatmap = GaussianCloud(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            colors=colors,
            opacities=self.opacities,
            provenance=self.provenance,
        )
        heatmap.to_ply(path, colors_linear=False)

    def to_gsplat(self) -> dict:
        """
        Convert to gsplat-compatible tensors.

        Returns dict with torch tensors:
        - quaternions: WXYZ order (gsplat convention)
        - colors: SH0 coefficients (NOT sRGB — gsplat decodes SH internally)
        - scales: linear (gsplat uses raw scales, no exp())
        - opacities: [0,1] (gsplat uses raw opacities, no sigmoid())
        """
        import torch

        # XYZW → WXYZ for gsplat
        quats_wxyz = np.stack(
            [self.quats[:, 3], self.quats[:, 0], self.quats[:, 1], self.quats[:, 2]],
            axis=-1,
        )

        # sRGB → SH0 coefficients (gsplat decodes SH internally)
        sh0_coeffs = (np.clip(self.colors, 0.0, 1.0) - 0.5) / SH_C0

        return {
            "means": torch.from_numpy(self.means).float(),
            "scales": torch.from_numpy(self.scales).float(),
            "quats": torch.from_numpy(quats_wxyz).float(),
            "colors": torch.from_numpy(sh0_coeffs).float(),
            "opacities": torch.from_numpy(self.opacities.squeeze()).float(),
        }

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to plain dict (XYZW quats, linear scales, raw opacities)."""
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "colors": self.colors,
            "opacities": self.opacities,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> GaussianCloud:
        """Create from plain dict."""
        return cls(
            means=d["means"],
            scales=d["scales"],
            quats=d["quats"],
            colors=d["colors"],
            opacities=d["opacities"],
            provenance=d.get("provenance"),
        )

    def merge(self, other: GaussianCloud) -> GaussianCloud:
        """Merge another cloud into this one, preserving provenance."""
        return GaussianCloud(
            means=np.concatenate([self.means, other.means]),
            scales=np.concatenate([self.scales, other.scales]),
            quats=np.concatenate([self.quats, other.quats]),
            colors=np.concatenate([self.colors, other.colors]),
            opacities=np.concatenate([self.opacities, other.opacities]),
            provenance=np.concatenate([self.provenance, other.provenance]),
        )

    def filter_by_provenance(self, source: GaussianSource) -> GaussianCloud:
        """Return a new cloud with only Gaussians of the given provenance."""
        mask = self.provenance == int(source)
        return GaussianCloud(
            means=self.means[mask],
            scales=self.scales[mask],
            quats=self.quats[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            provenance=self.provenance[mask],
        )

    def count_by_provenance(self) -> dict:
        """Count Gaussians per provenance type."""
        return {
            src.name: int(np.sum(self.provenance == int(src)))
            for src in GaussianSource
        }
