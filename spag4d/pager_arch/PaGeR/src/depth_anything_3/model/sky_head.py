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
"""Lightweight sky head: piggybacks on depth's post-fuse cache + late ViT levels, all detached."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkyBranchHead(nn.Module):
    """Sky head reading depth's neck cache + late ViT levels."""

    # Routing flags consumed by MultiDualDPT.forward (kwargs path).
    is_branch_head: bool = True
    needs_target_hw: bool = True
    needs_depth_decoder_feats: bool = True

    def __init__(
        self,
        token_dim: int = 3072,
        vit_levels: Sequence[int] = (2, 3),
        vit_mid: int = 256,
        vit_out: int = 32,
        depth_decoder_feat_dim: int = 128,
        sky_mid: int = 32,
        tower_blocks: int = 1,
        tower_dilation: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.vit_levels: Tuple[int, ...] = tuple(int(l) for l in vit_levels)
        self.vit_out = int(vit_out)
        self.depth_decoder_feat_dim = int(depth_decoder_feat_dim)

        # Per-level LN, concat, then a 2-stage MLP (single-shot Linear would over-compress).
        self.vit_norms = nn.ModuleList(
            [nn.LayerNorm(self.token_dim) for _ in self.vit_levels]
        )
        vit_in = len(self.vit_levels) * self.token_dim
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_in, int(vit_mid)),
            nn.GELU(),
            nn.Linear(int(vit_mid), self.vit_out),
        )

        # Tower runs at depth-neck resolution; ViT features upsampled and concatenated in.
        in_ch = self.depth_decoder_feat_dim + self.vit_out
        sky_mid = int(sky_mid)
        tower_blocks = max(1, int(tower_blocks))
        tower_dilation = max(1, int(tower_dilation))
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, sky_mid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        ]
        for _ in range(tower_blocks - 1):
            layers.append(nn.Conv2d(
                sky_mid, sky_mid, kernel_size=3, stride=1,
                padding=tower_dilation, dilation=tower_dilation,
            ))
            layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Conv2d(sky_mid, 1, kernel_size=1, stride=1, padding=0))
        self.fuse = nn.Sequential(*layers)

    def forward(
        self,
        feats: List[Tuple[torch.Tensor, torch.Tensor]],
        H: int,
        W: int,
        depth_decoder_feats: torch.Tensor,
        face_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """feats: list of (full_seq, cam_token) from ViT; returns (B, S, 1, H, W) sky logits."""
        max_lvl = max(self.vit_levels)
        assert len(feats) > max_lvl, (
            f"sky head expects at least {max_lvl + 1} feat levels, got {len(feats)}"
        )
        assert depth_decoder_feats is not None, (
            "SkyBranchHead requires depth_decoder_feats; check MultiDualDPT routing."
        )

        # Detach so sky losses can't reach the ViT.
        sample = feats[self.vit_levels[0]][0]
        assert sample.ndim == 4, f"expected (B, S, N, C), got {tuple(sample.shape)}"
        B, S, N, _ = sample.shape
        M = int(round(math.sqrt(N)))
        assert M * M == N, f"non-square patch grid: N={N}"

        normed = []
        for i, lvl in enumerate(self.vit_levels):
            x = feats[lvl][0].detach()
            normed.append(self.vit_norms[i](x))
        x = torch.cat(normed, dim=-1)
        x = self.vit_proj(x)

        vit_spatial = x.reshape(B * S, M, M, self.vit_out).permute(0, 3, 1, 2).contiguous()

        df = depth_decoder_feats.detach()
        BS, C_dec = B * S, df.shape[2]
        h_low, w_low = df.shape[-2], df.shape[-1]
        depth_spatial = df.reshape(BS, C_dec, h_low, w_low)
        assert C_dec == self.depth_decoder_feat_dim, (
            f"depth_decoder_feats has {C_dec} channels, head built for "
            f"{self.depth_decoder_feat_dim}. Override depth_decoder_feat_dim."
        )

        vit_up = F.interpolate(
            vit_spatial, size=(h_low, w_low), mode="bilinear", align_corners=True,
        )
        fused = torch.cat([depth_spatial, vit_up], dim=1)

        sky_logits = self.fuse(fused)
        sky_logits = F.interpolate(
            sky_logits, size=(int(H), int(W)), mode="bilinear", align_corners=True,
        )
        return sky_logits.reshape(B, S, *sky_logits.shape[1:])
