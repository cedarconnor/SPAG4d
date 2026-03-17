# spag4d/da360_model.py
"""
Wrapper for DA360 (Depth Anything 360) model.

DA360 is a fine-tuned Depth Anything V2 (ViT-Large) with circular padding
in the DPT decoder for seamless 360° depth estimation. Outputs scale-invariant
disparity (not metric depth), requiring inversion and scale normalization.

Key architecture details:
  - All Conv2d in DPT head replaced with ERPCircularConv2d
  - Horizontal: standard circular wrap (360° continuity)
  - Vertical/poles: roll by W/2 + flip (correct pole geometry)
  - Shift MLP on ViT class token: affine-invariant -> scale-invariant disparity
  - Input: 518x1036 ERP, ImageNet normalized

Reference: https://github.com/Insta360-Research-Team/DA360
Paper: "Depth Anything in 360: Towards Scale Invariance in the Wild"
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import hashlib


DA360_CONFIG = {
    # Weights are on Google Drive, downloaded via gdown or manual
    "gdrive_folder": "https://drive.google.com/drive/folders/1FMLWZfJ_IPKOa_cEbVqrq8_BRkl3oB_2",
    "filename": "DA360_large.pth",
    "sha256": None,
    "size_mb": 500,
}
DA360_CACHE_DIR = Path.home() / ".cache" / "spag4d"

# DA360 fixed input resolution (37*14=518 patches for ViT patch_size=14)
DA360_INPUT_H = 518
DA360_INPUT_W = 1036


class DA360Model:
    """
    Wrapper for DA360 depth estimation model.

    DA360 outputs scale-invariant disparity which is converted to depth
    via inversion and median-based scale normalization.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        device: torch.device = torch.device('cuda')
    ) -> 'DA360Model':
        """
        Load DA360 model from path or download weights.

        Args:
            model_path: Optional explicit path to weights
            device: Torch device to load to

        Returns:
            Loaded DA360Model instance
        """
        if model_path is None:
            model_path = cls._get_or_download_weights()

        # Import DA360 architecture
        try:
            from .da360_arch import build_da360_model
        except ImportError:
            raise ImportError(
                "DA360 architecture not found. Please set up DA360:\n"
                "1. Clone https://github.com/Insta360-Research-Team/DA360 "
                "into spag4d/da360_arch/DA360/\n"
                "2. Download DA360_large.pth from Google Drive\n"
            )

        model = build_da360_model()

        # Load checkpoint — DA360 stores metadata keys ('net', 'dinov2_encoder',
        # 'height', 'width') alongside model state_dict entries in a flat dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        metadata_keys = {'net', 'dinov2_encoder', 'height', 'width'}
        if isinstance(checkpoint, dict):
            model_state = model.state_dict()
            # Filter: skip metadata keys and only keep matching model keys
            filtered = {}
            for k, v in checkpoint.items():
                if k in metadata_keys:
                    continue
                # Strip 'module.' prefix if present (DataParallel)
                clean_k = k[7:] if k.startswith('module.') else k
                if clean_k in model_state:
                    filtered[clean_k] = v
            model.load_state_dict(filtered, strict=False)
            loaded = len(filtered)
            total = len(model_state)
            print(f"[DA360] Loaded {loaded}/{total} parameters")
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(device)
        return cls(model, device)

    @classmethod
    def _get_or_download_weights(cls) -> str:
        """Download weights or locate cached file."""
        DA360_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DA360_CACHE_DIR / DA360_CONFIG["filename"]

        if cache_path.exists():
            if cls._verify_checksum(cache_path):
                print(f"Using cached DA360 weights: {cache_path}")
                return str(cache_path)
            else:
                print("Cached DA360 weights corrupted, re-downloading...")
                cache_path.unlink()

        # Try gdown (Google Drive downloader)
        print(f"Downloading DA360 weights (~{DA360_CONFIG['size_mb']}MB)...")
        try:
            import gdown
            # DA360_large.pth file ID from Google Drive
            gdown.download(
                url=DA360_CONFIG["gdrive_folder"],
                output=str(DA360_CACHE_DIR),
                fuzzy=True,
                quiet=False,
            )
            if cache_path.exists():
                return str(cache_path)
        except ImportError:
            pass

        raise RuntimeError(
            f"DA360 weights not found at {cache_path}\n\n"
            "Download DA360_large.pth manually from:\n"
            f"  {DA360_CONFIG['gdrive_folder']}\n"
            f"Place it at: {cache_path}\n\n"
            "Or install gdown: pip install gdown"
        )

    @staticmethod
    def _verify_checksum(path: Path) -> bool:
        """Verify file SHA256 checksum."""
        if not DA360_CONFIG.get("sha256"):
            return True
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == DA360_CONFIG["sha256"]

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Predict depth from equirectangular image.

        DA360 outputs scale-invariant disparity; we invert to depth and
        normalize via median to get approximate metric scale.

        Args:
            image: RGB tensor [H, W, 3] or [B, H, W, 3], uint8 or [0,1] float
            return_mask: Ignored (DA360 doesn't produce masks)

        Returns:
            Tuple of (depth, None):
                - depth: [H, W] or [B, H, W] in approximate meters
                - mask: always None
        """
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)

        B, H, W, C = image.shape

        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # DA360 expects [B, C, H, W] with ImageNet normalization
        x = image.permute(0, 3, 1, 2).to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Resize to DA360's fixed input resolution (518x1036)
        x = F.interpolate(x, size=(DA360_INPUT_H, DA360_INPUT_W),
                          mode='bilinear', align_corners=True)

        # Run model
        output = self.model(x)

        # Extract disparity from output
        if isinstance(output, dict):
            disparity = output.get('pred_disp', None)
            if disparity is None:
                # Fallback to any available key
                disparity = next(iter(output.values()))
        elif isinstance(output, (tuple, list)):
            disparity = output[0]
        else:
            disparity = output

        # Remove channel dim [B, 1, H, W] -> [B, H, W]
        if disparity.dim() == 4:
            disparity = disparity.squeeze(1)

        # Disparity -> depth: depth = 1 / disparity
        eps = 1e-6
        depth = 1.0 / (disparity.abs() + eps)

        # Median-based scale normalization to approximate metric depth
        # Set median depth to ~5m (reasonable indoor/outdoor midpoint)
        for i in range(B):
            median_depth = depth[i].median()
            if median_depth > eps:
                depth[i] = depth[i] * (5.0 / median_depth)

        # Upsample to original resolution
        if depth.shape[-2] != H or depth.shape[-1] != W:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=True
            ).squeeze(1)

        if not is_batched:
            depth = depth.squeeze(0)

        return depth, None
