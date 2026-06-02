# flake8: noqa E501
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Modified 2026 by The PaGeR Authors.
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

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from addict import Dict

from depth_anything_3.model.dpt import _make_fusion_block, _make_scratch
from depth_anything_3.model.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)

# All head types ever recognised by this module (for old-checkpoint remapping).
_KNOWN_HEADS = ["depth", "normals", "sky"]


def _make_head_module(out_channels: Sequence[int], features: int, output_dim: int,
                      mid_features: int = 32,
                      valid_conv_padding: bool = False) -> nn.Module:
    """Per-head branch: layer_rn x4 + refinenet x4 + output_conv1 + output_conv2."""
    m = _make_scratch(list(out_channels), features, expand=False)
    m.refinenet1 = _make_fusion_block(features, valid_conv_padding=valid_conv_padding)
    m.refinenet2 = _make_fusion_block(features, valid_conv_padding=valid_conv_padding)
    m.refinenet3 = _make_fusion_block(features, valid_conv_padding=valid_conv_padding)
    m.refinenet4 = _make_fusion_block(features, has_residual=False,
                                      valid_conv_padding=valid_conv_padding)
    neck = features // 2
    m.output_conv1 = nn.Conv2d(features, neck, kernel_size=3, stride=1, padding=1)
    m.output_conv2 = nn.Sequential(
        nn.Conv2d(neck, mid_features, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(mid_features, output_dim, kernel_size=1, stride=1, padding=0),
    )
    return m


class DualDPT(nn.Module):
    """Multi-head DPT: shared norm/projects/resize_layers, independent fusion chain per head."""

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: List[str] = ["depth"],
        valid_conv_padding: bool = False,
        log_depth: bool = False,
        with_confidence: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()

        # Released PaGeR omits decoder global-PE, runs output_conv2 at face res,
        # and skips gradient checkpointing; those branches are excised below.
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_names = head_names
        self.valid_conv_padding = valid_conv_padding
        self.log_depth = log_depth
        self.with_confidence = with_confidence

        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        def _head_dims(head: str):
            if head == "normals":   return 3, 32
            if head == "depth" and with_confidence: return 2, 32  # depth + conf
            return 1, 32

        self.head_modules = nn.ModuleDict({
            head: _make_head_module(out_channels, features, *_head_dims(head),
                                    valid_conv_padding=valid_conv_padding)
            for head in head_names
        })

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if any(k.startswith(prefix + "scratch.") for k in state_dict):
            self._remap_scratch_to_head_modules(state_dict, prefix)
        self._copy_depth_weights_to_missing_heads(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _remap_scratch_to_head_modules(self, state_dict: dict, prefix: str) -> None:
        """Remap legacy scratch.* keys to head_modules.<head>.*, broadcasting depth weights to siblings."""
        scratch_prefix = prefix + "scratch."

        # Oldest format: scratch.output_conv2.* → scratch.depth_output_conv2.*
        for k in list(state_dict.keys()):
            if k.startswith(scratch_prefix + "output_conv2."):
                new_k = scratch_prefix + "depth_output_conv2." + k[len(scratch_prefix + "output_conv2."):]
                if new_k not in state_dict:
                    state_dict[new_k] = state_dict.pop(k)
                else:
                    state_dict.pop(k)

        shared: dict = {}
        per_head_conv2: dict = {}

        for k in list(state_dict.keys()):
            if not k.startswith(scratch_prefix):
                continue
            suffix = k[len(scratch_prefix):]

            matched_head = None
            for h in _KNOWN_HEADS:
                hp = f"{h}_output_conv2."
                if suffix.startswith(hp):
                    matched_head = h
                    conv2_suffix = suffix[len(hp):]
                    per_head_conv2.setdefault(h, {})[conv2_suffix] = state_dict.pop(k)
                    break

            if matched_head is None:
                shared[suffix] = state_dict.pop(k)

        # Broadcast shared weights to every active head.
        for head in self.head_names:
            hp = prefix + f"head_modules.{head}."
            for suffix, tensor in shared.items():
                state_dict[hp + suffix] = tensor.clone()

        depth_conv2 = per_head_conv2.get("depth", {})

        # Pre-slice depth's 2-channel final conv to 1ch for use as sibling init.
        depth_conv2_1ch: dict = {}
        for conv2_suffix, tensor in depth_conv2.items():
            if conv2_suffix in ("2.weight", "2.bias") and tensor.shape[0] > 1:
                depth_conv2_1ch[conv2_suffix] = tensor[:1]
            else:
                depth_conv2_1ch[conv2_suffix] = tensor

        for head in self.head_names:
            hp = prefix + f"head_modules.{head}."
            if head in per_head_conv2:
                for conv2_suffix, tensor in per_head_conv2[head].items():
                    state_dict[hp + f"output_conv2.{conv2_suffix}"] = tensor
            else:
                for conv2_suffix, tensor in depth_conv2_1ch.items():
                    w = tensor.clone()
                    if head == "normals" and conv2_suffix == "2.weight":
                        w = w.repeat(3, 1, 1, 1)
                    elif head == "normals" and conv2_suffix == "2.bias":
                        w = w.repeat(3)
                    state_dict[hp + f"output_conv2.{conv2_suffix}"] = w

        # Slice depth's 2-ch final conv down to model's expected channel count.
        if "depth" in self.head_modules:
            wk = prefix + "head_modules.depth.output_conv2.2.weight"
            bk = prefix + "head_modules.depth.output_conv2.2.bias"
            if wk in state_dict:
                target = self.head_modules["depth"].output_conv2[2].weight.shape
                w = state_dict[wk]
                if w.shape[0] > target[0]:
                    state_dict[wk] = w[:target[0]]
            if bk in state_dict:
                target = self.head_modules["depth"].output_conv2[2].bias.shape
                b = state_dict[bk]
                if b.shape[0] > target[0]:
                    state_dict[bk] = b[:target[0]]

    def _copy_depth_weights_to_missing_heads(self, state_dict: dict, prefix: str) -> None:
        """Seed missing heads with depth weights when adding a head to a single-head checkpoint."""
        if "depth" not in self.head_names:
            return
        depth_prefix = prefix + "head_modules.depth."
        depth_weights = {
            k[len(depth_prefix):]: state_dict[k]
            for k in state_dict if k.startswith(depth_prefix)
        }
        if not depth_weights:
            return

        for head in self.head_names:
            if head == "depth":
                continue
            head_prefix = prefix + f"head_modules.{head}."
            if any(k.startswith(head_prefix) for k in state_dict):
                continue

            for suffix, tensor in depth_weights.items():
                w = tensor.clone()
                # Depth's 2-ch final conv: use only the depth channel as sibling init.
                if suffix in ("output_conv2.2.weight", "output_conv2.2.bias") and w.shape[0] > 1:
                    w = w[:1]
                if head == "normals":
                    if suffix == "output_conv2.2.weight":
                        w = w.repeat(3, 1, 1, 1)
                    elif suffix == "output_conv2.2.bias":
                        w = w.repeat(3)

                state_dict[head_prefix + suffix] = w

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
        face_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        if chunk_size is None or chunk_size >= S:
            return self._forward_impl(feats, H, W, patch_start_idx, S=S, face_ids=face_ids)
        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dicts.append(self._forward_impl(
                [feat[s0:s1] for feat in feats], H, W, patch_start_idx, S=S, face_ids=face_ids,
            ))
        out_dict = {
            k: torch.cat([d[k] for d in out_dicts], dim=0)
            for k in out_dicts[0].keys()
        }
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        S: int = 6,
        face_ids: torch.Tensor = None,
    ) -> dict:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H, face_ids=face_ids)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        out_dict = {}
        for head in self.head_names:
            head_m = self.head_modules[head]

            fused, refinenet_outs = self._fuse_head(resized_feats, head_m)

            # Cache post-fuse and refinenet outputs for cross-head consumers (sky, scale).
            # Consumers detach; we keep depth's autograd graph intact for its own loss.
            if not hasattr(self, "_last_fused_per_head") or self._last_fused_per_head is None:
                self._last_fused_per_head = {}
            self._last_fused_per_head[head] = fused
            if not hasattr(self, "_last_refinenet_outs_per_head") or self._last_refinenet_outs_per_head is None:
                self._last_refinenet_outs_per_head = {}
            self._last_refinenet_outs_per_head[head] = refinenet_outs

            # Upsample to face res, then run output_conv2 on the full-res feature.
            fused = custom_interpolate(
                fused, (h_out, w_out), mode="bilinear",
                align_corners=True, valid_conv_padding=self.valid_conv_padding,
            )

            if self.pos_embed:
                fused = self._add_pos_embed(fused, W, H)

            head_x = head_m.output_conv2(fused)

            if head == "depth" and self.with_confidence:
                # 2-ch depth output: ch0=depth, ch1=conf.
                depth_logit = head_x[:, :1]
                conf_logit  = head_x[:, 1:2]
                if not self.log_depth:
                    depth_logit = self._apply_activation_single(depth_logit, self.activation)
                depth_conf = self._apply_activation_single(conf_logit, self.conf_activation)
                out_dict["depth"]      = depth_logit.reshape(B // S, S, *depth_logit.shape[1:])
                out_dict["depth_conf"] = depth_conf.reshape(B // S, S, *depth_conf.shape[1:])
                continue

            if not self.log_depth and head == "depth":
                head_x = self._apply_activation_single(head_x, self.activation)

            out_dict[head] = head_x.reshape(B // S, S, *head_x.shape[1:])

        return out_dict

    def _fuse_head(
        self, feats: List[torch.Tensor], head_m: nn.Module,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Pyramid fusion → (post-output_conv1 feature, [out_4..out_1] refinenet cache)."""
        l1, l2, l3, l4 = feats
        refinenet_outs: List[torch.Tensor] = []

        l1_rn = head_m.layer1_rn(l1)
        l2_rn = head_m.layer2_rn(l2)
        l3_rn = head_m.layer3_rn(l3)
        l4_rn = head_m.layer4_rn(l4)
        out_4 = head_m.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out_3 = head_m.refinenet3(out_4, l3_rn, size=l2_rn.shape[2:])
        out_2 = head_m.refinenet2(out_3, l2_rn, size=l1_rn.shape[2:])
        out_1 = head_m.refinenet1(out_2, l1_rn)
        refinenet_outs = [out_4, out_3, out_2, out_1]
        fused = head_m.output_conv1(out_1)

        return fused, refinenet_outs

    def _add_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1, **_ignored,
    ) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        x = x + pe
        return x

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":      return torch.exp(x)
        if act == "expm1":    return torch.expm1(x)
        if act == "expp1":    return torch.exp(x) + 1
        if act == "relu":     return torch.relu(x)
        if act == "sigmoid":  return torch.sigmoid(x)
        if act == "softplus": return torch.nn.functional.softplus(x)
        if act == "tanh":     return torch.tanh(x)
        return x


class MultiDualDPT(nn.Module):
    """Independent DualDPT (and scalar/branch) heads sharing only the upstream ViT."""

    def __init__(self, heads: dict, detach_heads=None):
        # detach_heads=None → detach every head's ViT features (safe default).
        super().__init__()
        self.heads = nn.ModuleDict(heads)
        self.detach_heads = None if detach_heads is None else frozenset(detach_heads)

    @staticmethod
    def _detach_feats(feats):
        return tuple((f0.detach(), f1.detach()) for f0, f1 in feats)

    def _route_feats(self, name, feats):
        if self.detach_heads is None or name in self.detach_heads:
            return self._detach_feats(feats)
        return feats

    def forward(self, feats, H, W, patch_start_idx, chunk_size=8, face_ids=None,
                skip_heads=None):
        skip = frozenset(skip_heads) if skip_heads else frozenset()
        out = {}
        for name, head in self.heads.items():
            if name in skip:
                continue
            head_feats = self._route_feats(name, feats)

            # Kwargs-dispatch for ScaleHead / SkyBranchHead; depth-cache flags route
            # depth's post-fuse and refinenet caches to scale/sky.
            if (getattr(head, "is_scalar_head", False)
                    or getattr(head, "is_branch_head", False)):
                kwargs = {"face_ids": face_ids}
                if getattr(head, "needs_target_hw", False):
                    kwargs["H"] = H
                    kwargs["W"] = W
                if getattr(head, "needs_depth_decoder_feats", False):
                    depth_head = self.heads["depth"] if "depth" in self.heads else None
                    cache = getattr(depth_head, "_last_fused_per_head", None) if depth_head is not None else None
                    fused = cache.get("depth") if cache else None
                    if fused is None:
                        raise RuntimeError(
                            f"Head '{name}' needs depth_decoder_feats but the "
                            "depth head's `_last_fused_per_head['depth']` is "
                            "missing. Ensure depth runs first and DualDPT "
                            "caching is active."
                        )
                    if "depth" not in out:
                        raise RuntimeError(
                            f"Head '{name}' needs depth_decoder_feats but the "
                            "depth output is missing — can't infer (B, S)."
                        )
                    BS = fused.shape[0]
                    B, S = out["depth"].shape[0], out["depth"].shape[1]
                    assert BS == B * S, (
                        f"depth fused features shape {tuple(fused.shape)} not "
                        f"compatible with depth out (B={B}, S={S})"
                    )
                    kwargs["depth_decoder_feats"] = fused.reshape(B, S, *fused.shape[1:])
                if getattr(head, "needs_si_pyramid_feats", False):
                    depth_head = self.heads["depth"] if "depth" in self.heads else None
                    cache = getattr(depth_head, "_last_refinenet_outs_per_head", None) if depth_head is not None else None
                    refinenet_outs = cache.get("depth") if cache else None
                    if refinenet_outs is None:
                        raise RuntimeError(
                            f"Head '{name}' needs si_pyramid_feats but the "
                            "depth head's `_last_refinenet_outs_per_head["
                            "'depth']` cache is missing. Ensure depth runs "
                            "first and DualDPT caching is active."
                        )
                    if "depth" not in out:
                        raise RuntimeError(
                            f"Head '{name}' needs si_pyramid_feats but the "
                            "depth output is missing — can't infer (B, S)."
                        )
                    B, S = out["depth"].shape[0], out["depth"].shape[1]
                    kwargs["si_pyramid_feats"] = [
                        f.reshape(B, S, *f.shape[1:]) for f in refinenet_outs
                    ]
                result = head(head_feats, **kwargs)
                # Dicts merge into `out`; tensors land at `out[name]`.
                if isinstance(result, dict):
                    # Direct-metric path: derive scalar pred_dict["scale"] from
                    # median(log_metric_pred − log_SI) so legacy consumers still see it.
                    if ("scale_log_metric_field" in result
                            and "scale" not in result):
                        log_metric_pred = result["scale_log_metric_field"]
                        if "depth" not in out:
                            raise RuntimeError(
                                f"Head '{name}' emitted "
                                "'scale_log_metric_field' but the depth "
                                "output is missing — cannot derive scale "
                                "scalar without log_SI."
                            )
                        log_si = out["depth"]
                        # Pool SI to scale head's downsampled grid when downsample_factor > 1.
                        if log_si.shape[-2:] != log_metric_pred.shape[-2:]:
                            B_, S_ = log_si.shape[0], log_si.shape[1]
                            log_si_4d = log_si.reshape(B_ * S_, 1, *log_si.shape[-2:])
                            log_si_4d = nn.functional.adaptive_avg_pool2d(
                                log_si_4d, log_metric_pred.shape[-2:],
                            )
                            log_si = log_si_4d.reshape(B_, S_, 1, *log_metric_pred.shape[-2:])
                        delta = (log_metric_pred - log_si)
                        Bk = delta.shape[0]
                        result["scale"] = (
                            delta.detach().reshape(Bk, -1)
                                 .median(dim=-1).values
                                 .unsqueeze(-1)
                        )
                    out.update(result)
                else:
                    out[name] = result
                continue

            head_out = head(head_feats, H, W, patch_start_idx, chunk_size, face_ids=face_ids)
            out.update(head_out)
        return Dict(out)
