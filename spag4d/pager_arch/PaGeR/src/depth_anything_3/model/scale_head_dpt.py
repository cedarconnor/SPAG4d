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
"""DPT-shaped scale head predicting log(metric) per anchor.

Mirrors the depth head's DualDPT body for direct weight transfer via init_from_depth.
Multi-stage adapter injection (post-fusion, zero-init projections) lets depth's pyramid
features feed the scale head while keeping it identity-on-SI at t=0. All depth signals
are detached. MultiDualDPT.forward derives the scalar pred_dict["scale"] post-hoc as
median(log_metric_pred − log_SI_pooled).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.model.dpt import _make_fusion_block, _make_scratch


class ScaleHeadDPT(nn.Module):
    """DPT-shaped scale head: depth-DPT mirror + adapter injection from depth pyramid."""

    # Routing markers consumed by MultiDualDPT.forward.
    is_scalar_head = True
    needs_si_pyramid_feats = True
    needs_depth_decoder_feats = False
    needs_target_hw = True

    def __init__(
        self,
        token_dim: int = 3072,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        features: int = 256,
        patch_size: int = 14,
        # Stages (coarse→fine: 0=post-refinenet4, 3=post-refinenet1) where the
        # depth pyramid's refinenet outputs are adapter-injected into ours.
        si_inject_stages: Sequence[int] = (0, 1, 2),
        si_proj_kernel: int = 1,
        with_confidence: bool = True,
        conf_activation: str = "expp1",
        mid_features: int = 32,
        # Output spatial downsample relative to face resolution; F>1 saves output_conv2 cost.
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.out_channels = tuple(int(c) for c in out_channels)
        self.features = int(features)
        self.patch_size = int(patch_size)
        # Sort + dedupe; cache the set form for cheap stage-membership tests.
        self.si_inject_stages: List[int] = sorted(set(int(s) for s in si_inject_stages))
        for s in self.si_inject_stages:
            assert 0 <= s < 4, (
                f"si_inject_stages indices must be in [0, 4), got {s}. "
                "0 = post-refinenet4 (coarsest), 3 = post-refinenet1 (finest)."
            )
        self.si_inject_stages_set = frozenset(self.si_inject_stages)
        self.si_proj_kernel = int(si_proj_kernel)
        self.with_confidence = bool(with_confidence)
        self.conf_activation = str(conf_activation)
        self.mid_features = int(mid_features)
        self.downsample_factor = max(1, int(downsample_factor))
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(self.token_dim)
        self.projects = nn.ModuleList([
            nn.Conv2d(self.token_dim, oc, kernel_size=1, stride=1, padding=0)
            for oc in self.out_channels
        ])
        # 4× / 2× / id / ½× ladder (matches DualDPT).
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(self.out_channels[0], self.out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(self.out_channels[1], self.out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(self.out_channels[3], self.out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        scratch = _make_scratch(list(self.out_channels), self.features, expand=False)
        self.layer1_rn = scratch.layer1_rn
        self.layer2_rn = scratch.layer2_rn
        self.layer3_rn = scratch.layer3_rn
        self.layer4_rn = scratch.layer4_rn

        self.refinenet1 = _make_fusion_block(self.features)
        self.refinenet2 = _make_fusion_block(self.features)
        self.refinenet3 = _make_fusion_block(self.features)
        self.refinenet4 = _make_fusion_block(self.features, has_residual=False)

        neck = self.features // 2
        self.output_conv1 = nn.Conv2d(self.features, neck, kernel_size=3,
                                      stride=1, padding=1)
        self.neck = neck

        # Zero-init adapter projections: proj(d) = 0 at t=0 → identity injection.
        self.si_pyramid_proj = nn.ModuleList()
        pad = self.si_proj_kernel // 2
        for _ in self.si_inject_stages:
            proj = nn.Conv2d(
                self.features, self.features,
                kernel_size=self.si_proj_kernel, padding=pad,
            )
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.si_pyramid_proj.append(proj)

        # Tail shape matches DualDPT so init_from_depth can copy in bulk.
        # Channels: ch0=log(metric), ch1=conf_logit when with_confidence.
        tail_out_dim = 2 if self.with_confidence else 1
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(neck, self.mid_features, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.mid_features, tail_out_dim,
                      kernel_size=1, stride=1, padding=0),
        )

        self.body_initialised_from_depth = False

    @torch.no_grad()
    def init_from_depth(self, depth_dualdpt) -> None:
        """Copy DPT-body + output_conv2 weights from a trained depth DualDPT.

        Combined with zero-init adapters, the head acts as identity-on-SI at t=0.
        """
        copied: list[str] = []
        skipped: list[str] = []

        def _copy(dst: nn.Module, src: nn.Module, name: str) -> None:
            try:
                dst.load_state_dict(src.state_dict())
                copied.append(name)
            except Exception as e:
                skipped.append(f"{name} ({e.__class__.__name__})")

        _copy(self.norm, depth_dualdpt.norm, "norm")
        _copy(self.projects, depth_dualdpt.projects, "projects")
        _copy(self.resize_layers, depth_dualdpt.resize_layers, "resize_layers")

        # nn.ModuleDict has no .get; guard with `in`.
        depth_mod = (
            depth_dualdpt.head_modules["depth"]
            if "depth" in depth_dualdpt.head_modules else None
        )
        if depth_mod is not None:
            _copy(self.layer1_rn, depth_mod.layer1_rn, "layer1_rn")
            _copy(self.layer2_rn, depth_mod.layer2_rn, "layer2_rn")
            _copy(self.layer3_rn, depth_mod.layer3_rn, "layer3_rn")
            _copy(self.layer4_rn, depth_mod.layer4_rn, "layer4_rn")
            _copy(self.refinenet1, depth_mod.refinenet1, "refinenet1")
            _copy(self.refinenet2, depth_mod.refinenet2, "refinenet2")
            _copy(self.refinenet3, depth_mod.refinenet3, "refinenet3")
            _copy(self.refinenet4, depth_mod.refinenet4, "refinenet4")
            _copy(self.output_conv1, depth_mod.output_conv1, "output_conv1")

            # Bulk-copy when shapes match; slice-copy when channel layouts differ.
            src = getattr(depth_mod, "output_conv2", None)
            if src is None:
                skipped.append("output_conv2 (depth has none)")
            else:
                src_final_w = src[2].weight
                tgt_final_w = self.output_conv2[2].weight
                if (src[0].weight.shape == self.output_conv2[0].weight.shape
                        and src_final_w.shape == tgt_final_w.shape):
                    try:
                        self.output_conv2.load_state_dict(src.state_dict())
                        copied.append("output_conv2 (full)")
                    except Exception as e:
                        skipped.append(f"output_conv2 ({e.__class__.__name__})")
                else:
                    try:
                        self.output_conv2[0].load_state_dict(src[0].state_dict())
                        copied.append("output_conv2.0 (3x3 mid)")
                    except Exception as e:
                        skipped.append(f"output_conv2.0 ({e.__class__.__name__})")
                    src_b = src[2].bias
                    tgt_b = self.output_conv2[2].bias
                    n_copy = min(src_final_w.shape[0], tgt_final_w.shape[0])
                    if src_final_w.shape[1] == tgt_final_w.shape[1]:
                        tgt_final_w.data[:n_copy].copy_(src_final_w.data[:n_copy])
                        tgt_b.data[:n_copy].copy_(src_b.data[:n_copy])
                        copied.append(f"output_conv2.2 (sliced {n_copy} channels)")
                    else:
                        skipped.append(
                            f"output_conv2.2 ({src_final_w.shape[1]}!={tgt_final_w.shape[1]} mid)"
                        )

        self.body_initialised_from_depth = bool(copied)
        if skipped:
            print(
                f"ScaleHeadDPT.init_from_depth: copied {len(copied)} modules; "
                f"skipped {skipped}"
            )

    def _resize_levels(
        self, feats: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """ViT levels → resized feature pyramid (mirrors DualDPT._forward_impl)."""
        B, S, N, C = feats[0][0].shape
        M = int(round(math.sqrt(N)))
        assert M * M == N, f"non-square patch grid: N={N}"
        BS = B * S

        resized: List[torch.Tensor] = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][0]                                  # (B, S, M*M, C)
            x = x.reshape(BS, M * M, C)
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(BS, C, M, M)
            x = self.projects[stage_idx](x)
            x = self.resize_layers[stage_idx](x)
            resized.append(x)
        return resized

    def _inject_si(
        self,
        x: torch.Tensor,
        si_pyramid_feats: List[torch.Tensor] | None,
        stage_idx: int,
        B: int, S: int,
    ) -> torch.Tensor:
        """x + proj(detach(d)) at stage stage_idx; no-op if cache absent or stage not selected."""
        if (si_pyramid_feats is None
                or stage_idx not in self.si_inject_stages_set):
            return x
        proj_idx = self.si_inject_stages.index(stage_idx)
        si_feat = si_pyramid_feats[stage_idx].detach()
        si_feat = si_feat.reshape(B * S, *si_feat.shape[2:])
        if si_feat.shape[-2:] != x.shape[-2:]:
            si_feat = F.interpolate(
                si_feat, size=x.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        return x + self.si_pyramid_proj[proj_idx](si_feat)

    def _fuse_pyramid(
        self,
        resized: List[torch.Tensor],
        si_pyramid_feats: List[torch.Tensor] | None,
        B: int, S: int,
    ) -> torch.Tensor:
        """DPT fusion: each refinenet, then adapter inject from matching depth stage."""
        l1, l2, l3, l4 = resized
        l1_rn = self.layer1_rn(l1)
        l2_rn = self.layer2_rn(l2)
        l3_rn = self.layer3_rn(l3)
        l4_rn = self.layer4_rn(l4)

        out = self.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self._inject_si(out, si_pyramid_feats, stage_idx=0, B=B, S=S)

        out = self.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self._inject_si(out, si_pyramid_feats, stage_idx=1, B=B, S=S)

        out = self.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self._inject_si(out, si_pyramid_feats, stage_idx=2, B=B, S=S)

        out = self.refinenet1(out, l1_rn)
        out = self._inject_si(out, si_pyramid_feats, stage_idx=3, B=B, S=S)

        out = self.output_conv1(out)
        return out

    @staticmethod
    def _apply_conf_activation(x: torch.Tensor, kind: str) -> torch.Tensor:
        kind = kind.lower()
        if kind == "expp1":   return torch.exp(x) + 1.0
        if kind == "softplus": return torch.nn.functional.softplus(x) + 1.0
        if kind == "sigmoid": return torch.sigmoid(x)
        return torch.exp(x) + 1.0   # default

    def forward(
        self,
        feats: List[Tuple[torch.Tensor, torch.Tensor]],
        face_ids: torch.Tensor | None = None,
        si_pyramid_feats: List[torch.Tensor] | None = None,
        H: int | None = None,
        W: int | None = None,
    ) -> dict:
        """Returns scale_log_metric_field (B, S, 1, H/F, W/F) and (optionally) _conf.

        si_pyramid_feats: depth head's coarse-to-fine refinenet outputs (None disables injection).
        H, W: face resolution, used to compute the anchor grid (H/F, W/F).
        ``MultiDualDPT.forward`` derives ``pred_dict["scale"]`` post-
            hoc by pooling SI depth to the same anchor grid and taking
            ``median(scale_log_metric_field − log_SI_pooled)``, so legacy
            consumers keep seeing a panorama scalar log-scale.
        """
        del face_ids

        B, S = feats[0][0].shape[0], feats[0][0].shape[1]

        resized = self._resize_levels(feats)
        fused = self._fuse_pyramid(resized, si_pyramid_feats, B, S)

        # Resize fused → (H/F, W/F) anchor grid so output_conv2 runs at the head's output res.
        assert H is not None and W is not None, (
            "ScaleHeadDPT requires H, W kwargs from MultiDualDPT routing"
        )
        target_h = max(1, H // self.downsample_factor)
        target_w = max(1, W // self.downsample_factor)
        fused_anchor = F.interpolate(
            fused, size=(target_h, target_w),
            mode="bilinear", align_corners=True,
        )
        out = self.output_conv2(fused_anchor)
        if self.with_confidence:
            log_metric = out[:, 0:1]
            conf_logit = out[:, 1:2]
            conf = self._apply_conf_activation(conf_logit, self.conf_activation)
            return {
                "scale_log_metric_field": log_metric.reshape(B, S, 1, target_h, target_w),
                "scale_log_metric_conf":  conf.reshape(B, S, 1, target_h, target_w),
            }
        log_metric = out
        return {
            "scale_log_metric_field": log_metric.reshape(B, S, 1, target_h, target_w),
        }
