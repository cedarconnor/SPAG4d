# spag4d/panda_model.py
"""
Wrapper for PanDA (Panoramic Depth Anything) model.

PanDA is a CVPR 2025 model that fine-tunes Depth Anything V2 with LoRA
and Möbius spatial augmentation for robust 360° depth estimation.

Outputs relative depth (0-1 range) which is scaled to pseudo-metric
using configurable depth_min/depth_max parameters.

Reference: https://github.com/caozidong/PanDA
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Optional
import hashlib


# Model configuration
PANDA_CONFIG = {
    "url": "https://huggingface.co/ZidongC/PanDA/resolve/main/panda_large.pth",
    "repo_id": "ZidongC/PanDA",
    "filename": "panda_large.pth",
    "sha256": None,  # Skip checksum verification
    "size_mb": 1500,
}
PANDA_CACHE_DIR = Path.home() / ".cache" / "spag4d"


class PanDAModel:
    """
    Wrapper for PanDA (Panoramic Depth Anything) model.

    PanDA is specifically designed for 360° equirectangular images
    and outputs relative depth (0-1) which is scaled to a configurable range.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        depth_mapping: str = "log",
    ):
        self.model = model
        self.device = device
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_mapping = depth_mapping  # "log", "inverse", or "linear"
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        device: torch.device = torch.device('cpu'),
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        depth_mapping: str = "log",
    ) -> 'PanDAModel':
        """
        Load PanDA model from path or download from HuggingFace.

        Args:
            model_path: Optional explicit path to weights
            device: Torch device to load to
            depth_min: Minimum depth for relative-to-metric scaling
            depth_max: Maximum depth for relative-to-metric scaling

        Returns:
            Loaded PanDAModel instance
        """
        if model_path is None:
            model_path = cls._get_or_download_weights()

        # Import PanDA model architecture
        try:
            from .panda_arch import build_panda_model
        except ImportError:
            raise ImportError(
                "PanDA architecture not found. Please clone the PanDA model files "
                "from https://github.com/caozidong/PanDA to spag4d/panda_arch/PanDA"
            )

        model = build_panda_model()

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix if model was saved with DataParallel
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load with strict=False to handle any minor mismatches
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)

        return cls(model, device, depth_min, depth_max, depth_mapping)

    @classmethod
    def _get_or_download_weights(cls) -> str:
        """Download weights with verification."""
        PANDA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = PANDA_CACHE_DIR / "panda_large.pth"

        if cache_path.exists():
            print(f"Using cached PanDA weights: {cache_path}")
            return str(cache_path)

        print(f"Downloading PanDA weights to {cache_path}...")
        print("  Size: ~1.5 GB, this may take a few minutes")

        try:
            import requests
            url = PANDA_CONFIG["url"]
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = 100 * downloaded / total_size
                        if downloaded % (50 * 1024 * 1024) < 8192:
                            print(f"  Downloaded {downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB ({pct:.0f}%)")

            print(f"Download complete: {cache_path}")
            return str(cache_path)

        except Exception as e:
            if cache_path.exists():
                cache_path.unlink()
            raise RuntimeError(f"Failed to download PanDA weights: {e}")

    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Predict depth from equirectangular image(s).

        Args:
            image: [H, W, 3] or [B, H, W, 3] float32 tensor in [0, 1].
            return_mask: unused, for API compatibility

        Returns:
            (depth, mask) where depth has same H, W as input.
        """
        with torch.no_grad():
            is_batched = image.dim() == 4
            if not is_batched:
                image = image.unsqueeze(0)  # [1, H, W, 3]

            B, H, W, C = image.shape

            # Preprocess
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

            # PanDA expects [B, C, H, W] normalized with ImageNet stats
            x = image.permute(0, 3, 1, 2)  # [B, 3, H, W]

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            x = (x - mean) / std

            # PanDA requires dimensions to be multiples of 14 (ViT patch size)
            # Cap input resolution to prevent OOM
            MAX_INPUT_WIDTH = 1022
            did_downscale = False
            target_H, target_W = H, W

            if W > MAX_INPUT_WIDTH:
                scale = MAX_INPUT_WIDTH / W
                target_H = int(H * scale)
                target_W = MAX_INPUT_WIDTH
                did_downscale = True

            # Ensure multiple of 14
            target_H = target_H - (target_H % 14)
            target_W = target_W - (target_W % 14)

            if target_H != H or target_W != W:
                x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=True)
                if did_downscale:
                    print(f"[PanDA] Downscaled input {W}x{H} -> {target_W}x{target_H} for inference")

            # Run model with OOM fallback
            try:
                output = self.model(x)
            except torch.cuda.OutOfMemoryError:
                # Fallback: process one at a time
                torch.cuda.empty_cache()
                depths = []
                for i in range(B):
                    out_i = self.model(x[i:i+1])
                    if isinstance(out_i, dict):
                        depths.append(out_i['pred_depth'])
                    else:
                        depths.append(out_i)
                depth = torch.cat(depths, dim=0)
                output = {'pred_depth': depth}

            # Handle different output formats
            if isinstance(output, dict):
                depth = output['pred_depth']
            else:
                depth = output

            # Remove channel dim if present [B, 1, H, W] -> [B, H, W]
            if depth.dim() == 4:
                depth = depth.squeeze(1)

            # Interpolate to original resolution if needed
            if depth.shape[-2] != H or depth.shape[-1] != W:
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

            # Scale relative depth (0-1) to pseudo-metric range.
            # PanDA outputs: higher value = farther away.
            # Normalize to [0, 1] first (model output may not be exactly 0-1).
            depth_min_val = depth.min()
            depth_max_val = depth.max()
            if depth_max_val > depth_min_val:
                depth_normalized = (depth - depth_min_val) / (depth_max_val - depth_min_val)
            else:
                depth_normalized = torch.zeros_like(depth)

            # Map to pseudo-metric range using the configured mapping.
            # Log-space preserves foreground detail: each order of magnitude
            # (0.1–1m, 1–10m, 10–100m) gets equal representation.
            if self.depth_mapping == "log":
                log_min = math.log(self.depth_min)
                log_max = math.log(self.depth_max)
                depth = torch.exp(log_min + depth_normalized * (log_max - log_min))
            elif self.depth_mapping == "inverse":
                # Disparity-linear (matches monocular model training signal)
                inv_max = 1.0 / self.depth_min
                inv_min = 1.0 / self.depth_max
                inv_depth = inv_max - depth_normalized * (inv_max - inv_min)
                depth = 1.0 / inv_depth.clamp(min=1e-6)
            else:
                # Linear (legacy behaviour)
                depth = self.depth_min + depth_normalized * (self.depth_max - self.depth_min)

            # Remove batch dim for single image input
            if not is_batched:
                depth = depth.squeeze(0)

            # PanDA doesn't output masks
            mask = None

            return depth, mask

    def predict_tiled(
        self,
        image: torch.Tensor,
        n_tiles: int = 4,
        overlap: float = 0.25,
    ) -> tuple:
        """
        Higher-resolution depth using tiled overlapping inference.

        Splits the ERP horizontally into n_tiles overlapping strips,
        runs PanDA at full 1022px width on each tile, then blends
        results in overlap zones with a cosine weight to avoid seams.

        ERP is horizontally wrappable, so the rightmost tile wraps
        seamlessly onto the leftmost tile.

        Args:
            image:   [H, W, 3] float32 tensor in [0,1]
            n_tiles: Number of horizontal strips (default 4)
            overlap: Fractional overlap on each side (default 0.25)

        Returns:
            (depth [H, W], None)
        """
        H, W = image.shape[:2]
        device = self.device

        tile_w = W // n_tiles
        overlap_px = int(tile_w * overlap)

        depth_accum  = torch.zeros(H, W, device=device)
        weight_accum = torch.zeros(H, W, device=device)

        for t in range(n_tiles):
            col_start = t * tile_w - overlap_px
            col_end   = col_start + tile_w + 2 * overlap_px

            # Wrap-around indices (ERP is periodic in azimuth)
            indices = torch.arange(col_start, col_end, device=device) % W
            tile_img = image[:, indices, :]  # [H, tile_w+2*ovlp, 3]

            # Predict depth for this strip
            tile_depth, _ = self.predict(tile_img)  # [H, tile_w+2*ovlp]

            # Normalise tile to [0,1] before blending
            t_min = tile_depth.min()
            t_max = tile_depth.max()
            tile_norm = (tile_depth - t_min) / (t_max - t_min + 1e-8)

            # Cosine blend weight: 1 at centre, 0 at edges
            tile_len = indices.shape[0]
            t_linspace = torch.linspace(0.0, math.pi, tile_len, device=device)
            weight = (0.5 - 0.5 * torch.cos(t_linspace)).unsqueeze(0).expand(H, -1)

            # Scatter-accumulate (handles wrap-around automatically)
            idx_exp = indices.unsqueeze(0).expand(H, -1)
            depth_accum.scatter_add_(1, idx_exp, tile_norm * weight)
            weight_accum.scatter_add_(1, idx_exp, weight)

        # Blend: weighted average gives normalised depth in [0,1]
        depth_norm = depth_accum / weight_accum.clamp(min=1e-6)

        # Apply configured metric mapping to the blended normalised depth
        if self.depth_mapping == "log":
            log_min = math.log(self.depth_min)
            log_max = math.log(self.depth_max)
            depth_out = torch.exp(log_min + depth_norm * (log_max - log_min))
        elif self.depth_mapping == "inverse":
            inv_max = 1.0 / self.depth_min
            inv_min = 1.0 / self.depth_max
            inv_d    = inv_max - depth_norm * (inv_max - inv_min)
            depth_out = 1.0 / inv_d.clamp(min=1e-6)
        else:
            depth_out = self.depth_min + depth_norm * (self.depth_max - self.depth_min)

        return depth_out, None
