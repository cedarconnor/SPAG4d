# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License 2.0; see the project LICENSE.

"""Pose-encoding helpers used by the DA3 camera encoder.

Trimmed to the forward path: rotation-matrix → quaternion conversion plus
the (T, quat, fov_h, fov_w) packing that ``CameraEnc`` reads.
"""

import torch
import torch.nn.functional as F


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw):
    """Pack ``(extrinsics, intrinsics, H×W)`` → ``(B, S, 9)`` pose encoding."""
    R = extrinsics[:, :, :3, :3]                 # (B, S, 3, 3)
    T = extrinsics[:, :, :3, 3]                  # (B, S, 3)
    quat = _mat_to_quat(R)                       # (B, S, 4) xyzw
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def _mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix → quaternion (xyzw, real-last, real-part >= 0)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1,
    )

    q_abs = _sqrt_pos(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12,         m02 - m20,         m10 - m01], dim=-1),
        torch.stack([m21 - m12,         q_abs[..., 1] ** 2, m10 + m01,         m02 + m20], dim=-1),
        torch.stack([m02 - m20,         m10 + m01,          q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01,         m20 + m02,          m21 + m12,          q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)

    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    out = candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]                  # rijk → xyzw
    return torch.where(out[..., 3:4] < 0, -out, out)  # real part ≥ 0


def _sqrt_pos(x: torch.Tensor) -> torch.Tensor:
    """``sqrt(max(0, x))`` with a zero subgradient at x=0."""
    positive = x > 0
    if torch.is_grad_enabled():
        out = torch.zeros_like(x)
        out[positive] = torch.sqrt(x[positive])
        return out
    return torch.where(positive, torch.sqrt(x), torch.zeros_like(x))
