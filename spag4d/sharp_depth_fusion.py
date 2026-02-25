# spag4d/sharp_depth_fusion.py
"""
Phase 1: Cubemap SHARP Depth Fusion for SPAG-4D.

Replaces Depth Pro with ML-Sharp's internal monodepth network.
Projects a 360° equirectangular panorama onto 6 cubemap faces, runs
SHARP's predictor on each face for metric depth, aligns the per-face depths
to the globally-consistent DAP/PanDA depth, then composites back to ERP
via feathered (edge-distance) blending.

Requires apple/ml-sharp:
    pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Reuse projection and blending logic originally written for Depth Pro
from .depth_pro_fusion import (
    ProjectionMode,
    project_erp_to_cubemap,
    align_depth_to_reference,
    composite_faces_to_erp,
)


class SharpDepthFusion:
    """
    Fuse Apple ML-Sharp face predictions with DAP panorama depth.

    Usage::

        fusion = SharpDepthFusion(device, face_size=1536)
        fusion.load_model()
        fused_depth, confidence = fusion.fuse(erp_image_np, dap_depth_np)

    The returned `confidence` map is suitable as the `confidence` argument to
    `DepthBlender.fuse()` in Phase 4, letting the Laplacian pyramid prefer
    SHARP where it is reliable and DAP elsewhere.
    """

    CACHE_DIR = Path.home() / ".cache" / "spag4d" / "sharp"
    SHARP_WEIGHTS_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    # SHARP native size is 1536
    SHARP_NATIVE_SIZE = 1536

    def __init__(
        self,
        device,
        face_size: int = 1536,
        fov_deg: float = 100.0,
        projection_mode: ProjectionMode = ProjectionMode.CUBEMAP,
        model_path: Optional[str] = None,
    ):
        self.device          = device
        self.face_size       = face_size
        self.fov_deg         = fov_deg
        self.projection_mode = projection_mode
        self.model_path      = model_path
        self._model          = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Lazy-load Apple ML-Sharp.
        """
        if self._model is not None:
            return

        try:
            from sharp.models import create_predictor, PredictorParams
        except ImportError as e:
            raise ImportError(
                "Apple SHARP is not installed. To enable high-quality refinement, install it with:\n"
                "`pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip`"
            ) from e

        path = model_path or self.model_path or self._get_or_download_weights()

        print(f"Loading SHARP weights from {path} for Depth Fusion...")
        params = PredictorParams()
        # Quality settings
        params.low_pass_filter_eps = 0.001
        self._model = create_predictor(params)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            state_dict = torch.load(path, map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

    def _get_or_download_weights(self) -> str:
        """Download SHARP weights if not cached."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self.CACHE_DIR / "sharp.pt"
        if cache_path.exists():
            return str(cache_path)
            
        print("Downloading SHARP weights...")
        try:
            # Try direct download from Apple CDN first
            import urllib.request
            import ssl
            # Create unverified context to avoid SSL errors
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(self.SHARP_WEIGHTS_URL, context=ctx) as response, open(cache_path, 'wb') as out_file:
                while True:
                    buffer = response.read(8192)
                    if not buffer:
                        break
                    out_file.write(buffer)
            print("\nDownload complete.")
            return str(cache_path)
        except Exception as e:
            print(f"Direct download failed: {e}. Trying fallback...")
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id="apple/Sharp",
                filename="sharp.pt",
                cache_dir=self.CACHE_DIR,
                local_dir=self.CACHE_DIR,
            )

    def fuse(
        self,
        erp_image: np.ndarray,   # (H, W, 3) uint8
        dap_depth: np.ndarray,   # (H, W) float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full fusion pipeline: project -> infer -> align -> composite.

        Args:
            erp_image:  (H, W, 3) uint8 equirectangular RGB panorama
            dap_depth:  (H, W) float32 globally-consistent depth (DAP/PanDA)

        Returns:
            fused_depth: (H, W) float32 in DAP metric space
            confidence:  (H, W) float32 in [0, 1] (1 = fully SHARP)
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded.  Call SharpDepthFusion.load_model() first."
            )

        if self.projection_mode == ProjectionMode.ICOSAHEDRON:
            warnings.warn(
                "ICOSAHEDRON projection is not yet implemented; "
                "falling back to CUBEMAP."
            )

        H, W = erp_image.shape[:2]

        # 1. Project ERP -> 6 cubemap faces
        # Note: we use self.face_size to construct the faces. If different
        # from SHARP's native 1536 resolution, it is handled internally via resizing.
        face_images, uv_maps = project_erp_to_cubemap(
            erp_image, face_size=self.face_size, fov_deg=self.fov_deg
        )

        # 2. Run SHARP on each face
        raw_depths = self._run_sharp(face_images)

        # 3. Align each face depth to DAP metric space
        aligned_depths: Dict[str, np.ndarray] = {}
        for face_name in raw_depths:
            aligned_depths[face_name] = align_depth_to_reference(
                raw_depths[face_name],
                uv_maps[face_name],
                dap_depth,
            )

        # 4. Composite back to ERP
        fused_depth, confidence = composite_faces_to_erp(
            aligned_depths, uv_maps, (H, W), dap_depth=dap_depth
        )
        return fused_depth, confidence

    def _run_sharp(
        self,
        face_images: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run Apple ML-Sharp's internal monodepth predictor on each face image.

        Returns:
            {face_name: (face_size, face_size) float32 depth in metres}
        """
        face_depths: Dict[str, np.ndarray] = {}
        
        # Focal length calculation based on FOV
        f_px = self.face_size / (2 * np.tan(np.radians(self.fov_deg / 2)))

        with torch.inference_mode():
            for face_name, face_img in face_images.items():
                if face_img.dtype != np.uint8:
                    face_img = face_img.clip(0, 255).astype(np.uint8)
                
                # Convert to FloatTensor
                face_tensor = torch.from_numpy(face_img).float() / 255.0
                face_tensor = face_tensor.to(self.device).permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]

                # Ensure resolution matches SHARP's SPN encoder requirements
                if face_tensor.shape[-1] != self.SHARP_NATIVE_SIZE:
                    face_tensor = F.interpolate(
                        face_tensor,
                        size=(self.SHARP_NATIVE_SIZE, self.SHARP_NATIVE_SIZE),
                        mode='bilinear',
                        align_corners=False,
                    )
                
                # Normalize disparity factor using native size
                disparity_factor_val = f_px / self.SHARP_NATIVE_SIZE
                disparity_factor = torch.tensor([disparity_factor_val], device=self.device, dtype=torch.float32)

                # Extract monodepth output
                monodepth_output = self._model.monodepth_model(face_tensor)
                disparity = monodepth_output.disparity
                
                # Convert disparity to unscaled depth
                disp_fac_expanded = disparity_factor[:, None, None, None]
                depth_t = disp_fac_expanded / disparity.clamp(min=1e-4, max=1e4) # [1, 1, H, W]
                
                # Use ML-Sharp's scaling aligner logic to refine local depth
                # Even without ground truth, it operates using local intermediate features.
                depth_t, _ = self._model.depth_alignment(depth_t, None, monodepth_output.decoder_features)
                
                # depth_t is [1, 2, H, W], where index 1 is the surface layer.
                # We only need the surface layer for depth fusion.
                
                # Resize back to requested output face size if it differs from native SHARP size
                if depth_t.shape[-1] != self.face_size:
                    depth_t = F.interpolate(
                        depth_t,
                        size=(self.face_size, self.face_size),
                        mode='bilinear',
                        align_corners=True
                    )
                    
                depth_t = depth_t[0, 1] # [H, W] surface layer
                
                face_depths[face_name] = depth_t.cpu().numpy().astype(np.float32)

        return face_depths
