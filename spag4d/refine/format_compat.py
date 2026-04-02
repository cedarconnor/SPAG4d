"""PLY format conversion between SPAG-4D and GSFix3D."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


def load_gaussians_from_ply(ply_path: str, device: str = "cuda"):
    """Load a SPAG-4D PLY into GSFix3D's GaussianModel.

    Handles format differences: adds missing normals, creates
    empty f_rest for SH degree 0, populates tensors directly.
    """
    from plyfile import PlyData
    from gs.gaussian_model import GaussianModel

    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    # Read position
    xyz = np.stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"]),
    ], axis=1).astype(np.float32)

    # Read SH DC coefficients
    f_dc = np.stack([
        np.asarray(vertex["f_dc_0"]),
        np.asarray(vertex["f_dc_1"]),
        np.asarray(vertex["f_dc_2"]),
    ], axis=1).astype(np.float32)

    # Read opacity, scale, rotation
    opacity = np.asarray(vertex["opacity"]).astype(np.float32)[:, np.newaxis]

    scale = np.stack([
        np.asarray(vertex["scale_0"]),
        np.asarray(vertex["scale_1"]),
        np.asarray(vertex["scale_2"]),
    ], axis=1).astype(np.float32)

    rot = np.stack([
        np.asarray(vertex["rot_0"]),
        np.asarray(vertex["rot_1"]),
        np.asarray(vertex["rot_2"]),
        np.asarray(vertex["rot_3"]),
    ], axis=1).astype(np.float32)

    n = xyz.shape[0]

    # Create GaussianModel and populate directly
    gaussians = GaussianModel(sh_degree=0)
    gaussians.active_sh_degree = 0

    # Set raw parameters (these are nn.Parameter on CUDA)
    gaussians._xyz = nn.Parameter(
        torch.from_numpy(xyz).cuda().requires_grad_(True)
    )
    gaussians._features_dc = nn.Parameter(
        torch.from_numpy(f_dc.reshape(n, 1, 3)).cuda().requires_grad_(True)
    )
    gaussians._features_rest = nn.Parameter(
        torch.zeros(n, 0, 3).cuda().requires_grad_(True)
    )
    gaussians._opacity = nn.Parameter(
        torch.from_numpy(opacity).cuda().requires_grad_(True)
    )
    gaussians._scaling = nn.Parameter(
        torch.from_numpy(scale).cuda().requires_grad_(True)
    )
    gaussians._rotation = nn.Parameter(
        torch.from_numpy(rot).cuda().requires_grad_(True)
    )

    # Initialize auxiliary tensors
    gaussians.max_radii2D = torch.zeros(n).cuda()
    gaussians.xyz_gradient_accum = torch.zeros(n, 1).cuda()
    gaussians.denom = torch.zeros(n, 1).cuda()

    logger.info(f"Loaded {n} Gaussians from {ply_path}")
    return gaussians


def save_gaussians_to_ply(gaussians, output_path: str):
    """Save GaussianModel to standard 3DGS PLY format."""
    if gaussians is None:
        logger.warning("save_gaussians_to_ply: gaussians is None, skipping")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(output_path)
    logger.info(f"Saved Gaussians to {output_path}")
