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

"""DA3 inference network: DinoV2 backbone + camera encoder + DPT head."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from depth_anything_3.cfg import create_object


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """DA3 inference network: ViT backbone + camera encoder + dense head."""

    PATCH_SIZE = 14

    def __init__(self, net, head, cam_enc=None, **_ignored):
        # **_ignored swallows multi-task yaml fields PaGeR doesn't use (cam_dec, gs_head, gs_adapter).
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_enc = (
            cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
        ) if cam_enc is not None else None

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        face_ids: Optional[torch.Tensor] = None,
        skip_heads=None,
    ) -> Dict[str, torch.Tensor]:
        """Backbone + dense head on a (B, N, 3, H, W) cubemap batch."""
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats = self.backbone(x, cam_token=cam_token, face_ids=face_ids)
        H, W = x.shape[-2], x.shape[-1]
        return self.head(feats, H, W, patch_start_idx=0, face_ids=face_ids, skip_heads=skip_heads)
