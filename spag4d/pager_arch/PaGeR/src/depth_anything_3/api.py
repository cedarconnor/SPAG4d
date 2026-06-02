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
"""Depth Anything 3 API: model loading, inference, export."""

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.model.utils.valid_conv_padding import set_valid_pad_conv
torch.backends.cudnn.benchmark = False

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """Depth Anything 3 model with HF Hub integration."""

    _commit_hash: str | None = None

    def __init__(self, model_name: str = "da3-large", **kwargs):
        super().__init__()
        self.model_name = model_name

        # Released PaGeR shares the DA3-Giant backbone; only head fields vary.
        self.config = load_config(MODEL_REGISTRY[self.model_name])

        model_cfg = kwargs["model_cfg"]
        self.config.head["head_names"] = model_cfg.modalities
        self.config.head["valid_conv_padding"] = model_cfg.valid_conv_padding
        self.config.head["log_depth"] = model_cfg.log_depth
        self.config.head["with_confidence"] = True
        self.model = create_object(self.config)

        if model_cfg.valid_conv_padding:
            set_valid_pad_conv(self.model.head)

        self.device = None

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        face_ids: torch.Tensor | None = None,
        skip_heads=None,
    ) -> dict[str, torch.Tensor]:
        """image: (B, N, 3, H, W); extrinsics/intrinsics accept legacy (N,...) or per-sample (B, N,...)."""
        ext_in = extrinsics if extrinsics.dim() == 4 else extrinsics[None]
        intr_in = intrinsics if intrinsics.dim() == 4 else intrinsics[None]
        ex_t_norm = self._normalize_extrinsics(ext_in.clone())
        prediction = self.model(
            image, ex_t_norm, intr_in, face_ids=face_ids, skip_heads=skip_heads,
        )
        return prediction
    

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm
    