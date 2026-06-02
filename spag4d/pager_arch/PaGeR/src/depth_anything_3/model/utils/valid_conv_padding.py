# Copyright 2026 The PaGeR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy edge-copy padding (kept for ablations); cube_resample_pad below supersedes it.
orderings = [
    [0, 1, 3, 4, 5],
    [1, 2, 0, 4, 5],
    [2, 3, 1, 4, 5],
    [3, 0, 2, 4, 5],
    [4, 1, 3, 2, 0],
    [5, 1, 3, 0, 2],
]
rotations = [
    [0, 0, 0, 0, 0],
    [0, 0, 0,-1, 1],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 1,-1],
    [0, 1,-1, 2, 0], 
    [0,-1, 1, 0, 2]
]

def _take_right(face, rot):
    if rot == 0:
        return face[..., :, 0]         
    elif rot == 1:
        return face[..., 0, :].flip(-1) 
    elif rot == 2:
        return face[..., :, -1].flip(-1)
    elif rot == -1:
        return face[..., -1, :]        

def _take_left(face, rot):
    if rot == 0:
        return face[..., :, -1]        
    elif rot == 1:
        return face[..., -1, :].flip(-1)
    elif rot == 2:
        return face[..., :, 0].flip(-1) 
    elif rot == -1:
        return face[..., 0, :]        

def _take_top(face, rot):
    if rot == 0:
        return face[..., -1, :]              
    elif rot == 1:
        return face[..., :, 0]               
    elif rot == 2:
        return face[..., 0, :].flip(-1)       
    elif rot == -1:
        return face[..., :, -1].flip(-1)      

def _take_bottom(face, rot):
    if rot == 0:
        return face[..., 0, :]               
    elif rot == 1:
        return face[..., :, -1]              
    elif rot == 2:
        return face[..., -1, :].flip(-1)      
    elif rot == -1:
        return face[..., :, 0].flip(-1)       

def valid_pad_conv_fn(x):
    assert x.ndim == 4
    N, C, H, W = x.shape
    # Sub-cubemap inputs fall back to reflect padding.
    if N % 6 != 0:
        return F.pad(x, [1, 1, 1, 1], mode='reflect')
    B = N // 6

    x_reshaped = x.view(B, 6, C, H, W)
    y = x.new_empty(B, 6, C, H+2, W+2)
    y[..., 1:-1, 1:-1] = x_reshaped

    for i in range(6):
        r_idx, l_idx, t_idx, b_idx = orderings[i][1:5]
        r_rot, l_rot, t_rot, b_rot = rotations[i][1:5]

        r_edge = _take_right (x_reshaped[:, r_idx], r_rot)
        l_edge = _take_left  (x_reshaped[:, l_idx], l_rot)
        t_edge = _take_top   (x_reshaped[:, t_idx], t_rot)
        b_edge = _take_bottom(x_reshaped[:, b_idx], b_rot)

        y[:, i, :, 1:-1, 0   ] = l_edge
        y[:, i, :, 1:-1, -1  ] = r_edge
        y[:, i, :, 0,     1:-1] = t_edge
        y[:, i, :, -1,    1:-1] = b_edge

        y[:, i, :, 0,  0 ] = 0.5*(y[:, i, :, 0, 1]   + y[:, i, :, 1, 0])
        y[:, i, :, 0, -1 ] = 0.5*(y[:, i, :, 0, -2]  + y[:, i, :, 1, -1])
        y[:, i, :, -1, 0 ] = 0.5*(y[:, i, :, -2, 0]  + y[:, i, :, -1, 1])
        y[:, i, :, -1,-1 ] = 0.5*(y[:, i, :, -2, -1] + y[:, i, :, -1, -2])

    return y.view(N, C, H+2, W+2)


# DreamCube-style spherical resample padding: for each padded pixel, raycast to the
# correct neighbour face and grid-sample. Required for face-local signals (z-depth).

_CUBE_EXTRINSICS_CACHE = None
_CUBE_PAD_GRID_CACHE = {}


def _get_default_extrinsics(device):
    """(6, 3, 3) world→camera rotations matching get_cubemap_intrinsics_extrinsics."""
    global _CUBE_EXTRINSICS_CACHE
    if _CUBE_EXTRINSICS_CACHE is None:
        face_configs = torch.tensor([
            (0.0, 0.0), (-90.0, 0.0), (180.0, 0.0), (90.0, 0.0), (0.0, -90.0), (0.0, 90.0),
        ], dtype=torch.float64)
        y = torch.deg2rad(face_configs[:, 0])
        p = torch.deg2rad(face_configs[:, 1])
        cy, sy = torch.cos(y), torch.sin(y)
        cp, sp = torch.cos(p), torch.sin(p)
        Ry = torch.zeros(6, 3, 3, dtype=torch.float64)
        Ry[:, 0, 0] = cy;   Ry[:, 0, 2] = sy
        Ry[:, 1, 1] = 1.0
        Ry[:, 2, 0] = -sy;  Ry[:, 2, 2] = cy
        Rx = torch.zeros(6, 3, 3, dtype=torch.float64)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = cp;   Rx[:, 1, 2] = -sp
        Rx[:, 2, 1] = sp;   Rx[:, 2, 2] = cp
        R = torch.bmm(Rx, Ry)
        R[R.abs() < 1e-10] = 0.0  # snap sin/cos(±90) jitter
        _CUBE_EXTRINSICS_CACHE = R.to(torch.float32)
    return _CUBE_EXTRINSICS_CACHE.to(device)


def _make_intrinsics(H, W, fov_deg, device):
    assert H == W, f"cube faces must be square, got ({H}, {W})"
    f = H / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    K = torch.tensor([[f, 0, W / 2.0],
                      [0, f, H / 2.0],
                      [0, 0, 1.0]], dtype=torch.float32, device=device)
    return K.unsqueeze(0).expand(6, -1, -1)


def _build_cube_pad_grid(H, W, padding, fov_deg, device):
    """Precompute the 3-D grid-sample grid (6, H_pad, W_pad, 3) and border mask."""
    key = (H, W, padding, fov_deg, str(device))
    cached = _CUBE_PAD_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    P = padding
    H_pad, W_pad = H + 2 * P, W + 2 * P

    R_all = _get_default_extrinsics(device)
    K_all = _make_intrinsics(H, W, fov_deg, device)

    # Sample pixel centres (k+0.5) — avoids argmax ties on cube corners.
    v_pix, u_pix = torch.meshgrid(
        torch.arange(H_pad, device=device, dtype=torch.float32) + 0.5 - P,
        torch.arange(W_pad, device=device, dtype=torch.float32) + 0.5 - P,
        indexing='ij',
    )
    ones = torch.ones_like(u_pix)

    # Source-face rays → world frame.
    ray_world_list = []
    for i in range(6):
        fx = K_all[i, 0, 0]; fy = K_all[i, 1, 1]
        cx = K_all[i, 0, 2]; cy = K_all[i, 1, 2]
        dx = (u_pix - cx) / fx
        dy = (v_pix - cy) / fy
        ray_cam = torch.stack([dx, dy, ones], dim=-1)
        ray_world = torch.einsum('ji,hwj->hwi', R_all[i], ray_cam)
        ray_world_list.append(ray_world)
    ray_world = torch.stack(ray_world_list, dim=0)

    # Target face = argmax of forward · ray (forward_j = R_j[2, :]).
    forwards = R_all[:, 2, :]
    dots = torch.einsum('kc,shwc->shwk', forwards, ray_world)
    face_j = torch.argmax(dots, dim=-1)

    R_j = R_all[face_j]
    K_j = K_all[face_j]
    ray_cam_j = torch.einsum('shwab,shwb->shwa', R_j, ray_world)
    z = ray_cam_j[..., 2:3].clamp(min=1e-6)
    pixel_j = torch.einsum('shwab,shwb->shwa', K_j, ray_cam_j / z)

    u_j = pixel_j[..., 0]
    v_j = pixel_j[..., 1]
    # align_corners=False normalisation; the +0.5 centre offset is already in u_j.
    u_norm = 2.0 * u_j / W - 1.0
    v_norm = 2.0 * v_j / H - 1.0
    face_z_norm = (2.0 * face_j.to(torch.float32) + 1.0) / 6.0 - 1.0
    grid = torch.stack([u_norm, v_norm, face_z_norm], dim=-1)

    mask = torch.ones(H_pad, W_pad, dtype=torch.bool, device=device)
    mask[P:-P, P:-P] = False

    _CUBE_PAD_GRID_CACHE[key] = (grid, mask)
    return grid, mask


def cube_resample_pad(x, padding, fov_deg=90.0):
    """DreamCube-style spherical resample padding for cubemaps stacked on the batch dim."""
    assert x.ndim == 4, f"expected 4-D (N, C, H, W), got {tuple(x.shape)}"
    N, C, H, W = x.shape
    P = int(padding)
    if P <= 0:
        return x
    # Sub-cubemap inputs fall back to reflect padding (no neighbour faces to stitch from).
    if N % 6 != 0:
        return F.pad(x, [P] * 4, mode='reflect')
    B = N // 6
    H_pad, W_pad = H + 2 * P, W + 2 * P
    device, dtype = x.device, x.dtype

    grid, mask = _build_cube_pad_grid(H, W, P, fov_deg, device)

    # 3-D grid_sample on a 6-slice volume; per-face z-centres collapse bilinear onto one slice.
    x_vol = x.view(B, 6, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    grid_b = grid.to(torch.float32).unsqueeze(0).expand(B, -1, -1, -1, -1)
    sampled = F.grid_sample(
        x_vol.to(torch.float32), grid_b,
        mode='bilinear', padding_mode='border', align_corners=False,
    )
    sampled = sampled.permute(0, 2, 1, 3, 4)

    # Reflect-pad baseline so the interior stays bit-identical.
    base = F.pad(x, [P] * 4, mode='reflect').view(B, 6, C, H_pad, W_pad)
    out = torch.where(mask.view(1, 1, 1, H_pad, W_pad), sampled.to(dtype), base)
    return out.reshape(N, C, H_pad, W_pad)


def make_cube_resample_pad_fn(padding=1, fov_deg=90.0):
    def _fn(x):
        return cube_resample_pad(x, padding=padding, fov_deg=fov_deg)
    return _fn


class PaddedConv2d(nn.Conv2d):
    def __init__(self, *args, pad_fn=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.pad_fn = pad_fn

    def forward(self, x):
        x = self.pad_fn(x)
        return F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=0,
            dilation=self.dilation, groups=self.groups
        )

    @classmethod
    def from_existing(cls, conv: nn.Conv2d, pad_fn):
        new = cls(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=0, dilation=conv.dilation,
            groups=conv.groups, bias=(conv.bias is not None),
            padding_mode="zeros", pad_fn=pad_fn
        )
        new.weight = conv.weight
        if conv.bias is not None:
            new.bias = conv.bias
        return new


def set_valid_pad_conv(module: nn.Module, fov_deg: float = 90.0):
    """Recursively replace every padded Conv2d with the DreamCube-padded variant."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            if child.kernel_size != (1, 1) and child.padding != (0, 0):
                P = int(child.padding[0])
                pad_fn = make_cube_resample_pad_fn(padding=P, fov_deg=fov_deg)
                setattr(module, name, PaddedConv2d.from_existing(child, pad_fn))
        else:
            set_valid_pad_conv(child, fov_deg=fov_deg)


