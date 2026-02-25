# spag4d/da3_model.py
"""
Wrapper for Depth Anything V3 (DA3) model.

DA3 is a unified depth estimation model from ByteDance-Seed that supports
monocular depth, multi-view depth, pose estimation, and Gaussian prediction.

For SPAG-4D we use DA3Metric-Large for metric depth output (real-world scale)
or DA3Mono-Large for relative depth (similar to PanDA).

Depth Anything V3 Integration
Install: pip install --no-deps https://github.com/ByteDance-Seed/depth-anything-3/archive/refs/heads/main.zip
Reference: https://github.com/ByteDance-Seed/Depth-Anything-3
"""

import torch
import numpy as np
from typing import Optional


# Model presets
DA3_MODELS = {
    "metric": "depth-anything/DA3METRIC-LARGE",
    "mono": "depth-anything/DA3MONO-LARGE",
    "large": "depth-anything/DA3-LARGE",
}


class DA3Model:
    """
    Wrapper for Depth Anything V3 with same predict() interface as DAP/PanDA.

    Uses DA3Metric-Large by default for metric depth output (real-world meters).
    Falls back to DA3Mono-Large (relative depth) if metric is unavailable.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        variant: str = "metric",
        depth_min: float = 0.1,
        depth_max: float = 100.0,
    ):
        self.model = model
        self.device = device
        self.variant = variant
        self.depth_min = depth_min
        self.depth_max = depth_max
        # DA3Metric outputs metric depth directly; DA3Mono outputs relative
        self.is_metric = variant == "metric"

    @classmethod
    def load(
        cls,
        variant: str = "metric",
        device: torch.device = torch.device('cuda'),
        depth_min: float = 0.1,
        depth_max: float = 100.0,
    ) -> 'DA3Model':
        """
        Load DA3 model from HuggingFace.

        Args:
            variant: Model variant - "metric" (real scale) or "mono" (relative)
            device: Torch device
            depth_min: Min depth for relative-to-metric scaling (mono variant only)
            depth_max: Max depth for relative-to-metric scaling (mono variant only)

        Returns:
            Loaded DA3Model instance
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Depth Anything V3 not installed. Install with:\n"
                '  pip install --no-deps https://github.com/ByteDance-Seed/depth-anything-3/archive/refs/heads/main.zip\n'
                "Or: pip install depth-anything-3"
            ) from exc

        model_id = DA3_MODELS.get(variant)
        if model_id is None:
            raise ValueError(f"Unknown DA3 variant '{variant}'. Choose from: {list(DA3_MODELS.keys())}")

        print(f"Loading DA3 model: {model_id}")
        model = DepthAnything3.from_pretrained(model_id)
        model = model.to(device=device)

        return cls(model, device, variant, depth_min, depth_max)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False,
        projection_mode: str = "equirectangular",
        base_model = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict depth from equirectangular image.

        Args:
            image: RGB image tensor [H, W, 3] uint8 or [0,1] float
            return_mask: Unused, kept for API compatibility

        Returns:
            Tuple of (depth, mask):
                - depth: [H, W] in meters (metric) or pseudo-metric (mono)
                - mask: Confidence mask [H, W] or None
        """
        H, W = image.shape[0], image.shape[1]

        # DA3 expects numpy arrays or PIL images, not tensors
        if image.dtype == torch.uint8:
            img_np = image.cpu().numpy()  # [H, W, 3] uint8
        else:
            img_np = (image.cpu().float() * 255).clamp(0, 255).byte().numpy()

        if projection_mode in ["cubemap", "icosahedral"]:
            return self._predict_projected(img_np, H, W, projection_mode, base_model=base_model, image_tensor=image)
        else:
            return self._predict_single(img_np, H, W)

    def _predict_single(
        self,
        img_np: np.ndarray,
        H: int,
        W: int,
        process_res_cap: int = 2048
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict depth for a single image array."""
        max_dim = max(H, W)
        process_res = min(max_dim, process_res_cap)
        
        prediction = self.model.inference(
            [img_np],
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )

        depth_np = prediction.depth[0]  # [H, W]
        depth = torch.from_numpy(depth_np).to(self.device)

        if depth.shape[0] != H or depth.shape[1] != W:
            import torch.nn.functional as F
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=True,
            ).squeeze()

        if not self.is_metric:
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth_normalized = (depth - d_min) / (d_max - d_min)
            else:
                depth_normalized = torch.zeros_like(depth)
            depth = self.depth_min + depth_normalized * (self.depth_max - self.depth_min)

        mask = None
        if prediction.conf is not None:
            conf_np = prediction.conf[0]  # [H, W]
            mask = torch.from_numpy(conf_np).to(self.device)
            if mask.shape[0] != H or mask.shape[1] != W:
                import torch.nn.functional as F
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True,
                ).squeeze()

        return depth, mask

    def _predict_projected(
        self,
        erp_img_np: np.ndarray,
        H: int,
        W: int,
        projection_mode: str,
        base_model = None,
        image_tensor = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict depth using projected faces aligned to a global low-res depth map."""
        from spag4d.projection import get_projector
        
        # 1. First, get a global "base" depth prediction to serve as scale/shift calibration.
        #    If base_model is provided (e.g. DAP), use it for a superior latlong anchor.
        #    Otherwise, cap the resolution lower (e.g. 1024) to keep DA3 fast for alignment.
        global_mask = None
        if base_model is not None and image_tensor is not None:
             global_depth, global_mask = base_model.predict(image_tensor)
        else:
             global_depth, global_mask = self._predict_single(erp_img_np, H, W, process_res_cap=1024)
        
        # 2. Get the appropriate projector and project the high-res ERP image
        # Determine an appropriate face size depending on the input resolution
        if projection_mode == "cubemap":
            # 6 faces: roughly 1/4 the equirectangular width
            face_size = max(512, W // 4)
        else: # icosahedral
            # 20 faces: roughly 1/5 the equirectangular width
            face_size = max(384, W // 5)
            
        projector = get_projector(projection_mode, face_size, self.device)
        faces_rgb_np = projector.project_erp_to_faces(erp_img_np)
        
        # Also project the global depth so we can align each face's depth to its region in the global map
        global_depth_np = global_depth.cpu().numpy()[..., np.newaxis] # [H, W, 1]
        # We need a float32 dummy "RGB" image for the projection utilities which expect 3 channels usually
        # But wait, projector.project_erp_to_faces expects np.ndarray. For our projector, it can handle 1 channel or 3
        # Icosahedral handles [H,W,C] gracefully. Cubemap Equirec2Cube also handles arbitrary channels? 
        # Actually DAP Equirec2Cube expects [H, W, 3] or [H, W, C]. Let's replicate depth to 3 channels to be safe.
        global_depth_3c = np.repeat(global_depth_np, 3, axis=-1)
        faces_global_depth_3c = projector.project_erp_to_faces(global_depth_3c)
        faces_global_depth = [f[..., 0] for f in faces_global_depth_3c]
        
        # 3. Predict high-res depth for each face
        raw_face_depths = []
        face_conf_tensors = []
        for i in range(len(faces_rgb_np)):
            face_rgb = faces_rgb_np[i]
            face_depth_pred, face_conf = self._predict_single(face_rgb, face_size, face_size)
            raw_face_depths.append(face_depth_pred.cpu().numpy())
            if face_conf is not None:
                face_conf_tensors.append(face_conf)

        # 3b. Alignment strategy depends on the model variant:
        #
        #  - "metric" (DA3Metric): each face already outputs real-world metres.
        #    Alignment to the global map would de-calibrate the scale (because the
        #    low-res global prediction has the same metric uncertainty).  Skip it.
        #
        #  - "mono" (DA3Mono / relative): depths are up-to-a-scale, so we fit a
        #    single global (scale, shift) across all faces to anchor them to the
        #    low-res prediction, ensuring inter-face continuity.

        if self.is_metric:
            # Metric: use per-face depths directly (already in metres)
            global_scale, global_shift = 1.0, 0.0
        else:
            # Mono: fit ONE global affine from combined face samples to the global depth
            all_src_samples = []
            all_ref_samples = []
            for i, face_depth_np in enumerate(raw_face_depths):
                h_f, w_f = face_depth_np.shape
                margin = int(h_f * 0.1)
                all_src_samples.append(face_depth_np[margin:-margin, margin:-margin].flatten())
                all_ref_samples.append(faces_global_depth[i][margin:-margin, margin:-margin].flatten())

            src_all = np.concatenate(all_src_samples)
            ref_all = np.concatenate(all_ref_samples)

            if len(src_all) > 100:
                A = np.vstack([src_all, np.ones(len(src_all))]).T
                try:
                    res = np.linalg.lstsq(A, ref_all, rcond=None)[0]
                    global_scale, global_shift = float(res[0]), float(res[1])
                except np.linalg.LinAlgError:
                    global_scale = 1.0
                    global_shift = float(np.mean(ref_all) - np.mean(src_all))
                global_scale = float(np.clip(global_scale, 0.1, 10.0))
            else:
                global_scale, global_shift = 1.0, 0.0

        # 3c. Apply uniform alignment to every face (scale=1, shift=0 for metric)
        aligned_depth_faces_tensors = []
        for face_depth_np in raw_face_depths:
            aligned = np.clip(face_depth_np * global_scale + global_shift, 0.01, None)
            aligned_depth_faces_tensors.append(
                torch.from_numpy(aligned).float().to(self.device)
            )

                
        # 4. Stitch aligned faces back to ERP
        stitched_depth = projector.reproject_to_erp(aligned_depth_faces_tensors, H, W)
        
        stitched_mask = None
        if face_conf_tensors and len(face_conf_tensors) == len(faces_rgb_np):
            stitched_mask = projector.reproject_to_erp(face_conf_tensors, H, W)
            
        return stitched_depth, stitched_mask
