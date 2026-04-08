"""Contains linear algebra related utility functions.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_matrices_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert batch of quaternions into rotations matrices.

    Args:
        quaternions: The quaternions convert to matrices.

    Returns:
        The rotations matrices corresponding to the (normalized) quaternions.
    """
    device = quaternions.device
    shape = quaternions.shape[:-1]

    quaternions = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    real_part = quaternions[..., 0]
    vector_part = quaternions[..., 1:]

    vector_cross = get_cross_product_matrix(vector_part)
    real_part = real_part[..., None, None]

    matrix_outer = vector_part[..., :, None] * vector_part[..., None, :]
    matrix_diag = real_part.square() * eyes(3, shape=shape, device=device)
    matrix_cross_1 = 2 * real_part * vector_cross
    matrix_cross_2 = vector_cross @ vector_cross

    return matrix_outer + matrix_diag + matrix_cross_1 + matrix_cross_2


def quaternions_from_rotation_matrices(matrices: torch.Tensor) -> torch.Tensor:
    """Convert batch of rotation matrices to quaternions.

    Args:
        matrices: The matrices to convert to quaternions.

    Returns:
        The quaternions corresponding to the rotation matrices.
    """
    if not matrices.shape[-2:] == (3, 3):
        raise ValueError(f"matrices have invalid shape {matrices.shape}")
    matrices = matrices.detach()
    original_shape = matrices.shape[:-2]
    matrices = matrices.reshape(-1, 3, 3)

    m00 = matrices[:, 0, 0]
    m01 = matrices[:, 0, 1]
    m02 = matrices[:, 0, 2]
    m10 = matrices[:, 1, 0]
    m11 = matrices[:, 1, 1]
    m12 = matrices[:, 1, 2]
    m20 = matrices[:, 2, 0]
    m21 = matrices[:, 2, 1]
    m22 = matrices[:, 2, 2]

    trace = m00 + m11 + m22
    eps = torch.finfo(matrices.dtype).eps
    quaternions = torch.empty((matrices.shape[0], 4), device=matrices.device, dtype=matrices.dtype)

    mask0 = trace > 0
    mask1 = (~mask0) & (m00 > m11) & (m00 > m22)
    mask2 = (~mask0) & (~mask1) & (m11 > m22)
    mask3 = ~(mask0 | mask1 | mask2)

    if mask0.any():
        scale = torch.sqrt(torch.clamp(trace[mask0] + 1.0, min=eps)) * 2.0
        quaternions[mask0, 0] = 0.25 * scale
        quaternions[mask0, 1] = (m21[mask0] - m12[mask0]) / scale
        quaternions[mask0, 2] = (m02[mask0] - m20[mask0]) / scale
        quaternions[mask0, 3] = (m10[mask0] - m01[mask0]) / scale

    if mask1.any():
        scale = torch.sqrt(torch.clamp(1.0 + m00[mask1] - m11[mask1] - m22[mask1], min=eps)) * 2.0
        quaternions[mask1, 0] = (m21[mask1] - m12[mask1]) / scale
        quaternions[mask1, 1] = 0.25 * scale
        quaternions[mask1, 2] = (m01[mask1] + m10[mask1]) / scale
        quaternions[mask1, 3] = (m02[mask1] + m20[mask1]) / scale

    if mask2.any():
        scale = torch.sqrt(torch.clamp(1.0 + m11[mask2] - m00[mask2] - m22[mask2], min=eps)) * 2.0
        quaternions[mask2, 0] = (m02[mask2] - m20[mask2]) / scale
        quaternions[mask2, 1] = (m01[mask2] + m10[mask2]) / scale
        quaternions[mask2, 2] = 0.25 * scale
        quaternions[mask2, 3] = (m12[mask2] + m21[mask2]) / scale

    if mask3.any():
        scale = torch.sqrt(torch.clamp(1.0 + m22[mask3] - m00[mask3] - m11[mask3], min=eps)) * 2.0
        quaternions[mask3, 0] = (m10[mask3] - m01[mask3]) / scale
        quaternions[mask3, 1] = (m02[mask3] + m20[mask3]) / scale
        quaternions[mask3, 2] = (m12[mask3] + m21[mask3]) / scale
        quaternions[mask3, 3] = 0.25 * scale

    quaternions = F.normalize(quaternions, dim=-1)
    quaternions = torch.where(quaternions[:, :1] < 0, -quaternions, quaternions)
    return quaternions.reshape(original_shape + (4,))


def get_cross_product_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Generate cross product matrix for vector exterior product."""
    if not vectors.shape[-1] == 3:
        raise ValueError("Only 3-dimensional vectors are supported")
    device = vectors.device
    shape = vectors.shape[:-1]
    unit_basis = eyes(3, shape=shape, device=device)
    # We compute the matrix by multiplying each column of unit_basis with the
    # corresponding vector.
    return torch.cross(vectors[..., :, None], unit_basis, dim=-2)


def eyes(
    dim: int, shape: tuple[int, ...], device: torch.device | str | None = None
) -> torch.Tensor:
    """Create batch of identity matrices."""
    return torch.eye(dim, device=device).broadcast_to(shape + (dim, dim)).clone()


def quaternion_product(q1, q2):
    """Compute dot product between two quaternions."""
    real_1 = q1[..., :1]
    real_2 = q2[..., :1]
    vector_1 = q1[..., 1:]
    vector_2 = q2[..., 1:]

    real_out = real_1 * real_2 - (vector_1 * vector_2).sum(dim=-1, keepdim=True)
    vector_out = real_1 * vector_2 + real_2 * vector_1 + torch.cross(vector_1, vector_2)
    return torch.concatenate([real_out, vector_out], dim=-1)


def quaternion_conj(q):
    """Get conjugate of a quaternion."""
    real = q[..., :1]
    vector = q[..., 1:]
    return torch.concatenate([real, -vector], dim=-1)


def project(u: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project tensor u to unit basis a."""
    unit_u = F.normalize(u, dim=-1)
    inner_prod = (unit_u * basis).sum(dim=-1, keepdim=True)
    return inner_prod * u
