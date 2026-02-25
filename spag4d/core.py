# spag4d/core.py
"""
Main SPAG4D class that orchestrates the conversion pipeline.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, Union
import time

from PIL import Image, ImageOps

# New quality modules (imported lazily to avoid hard dependencies at startup)
# scene_filter, adaptive_stride, depth_blend


@dataclass
class ConversionResult:
    """Result of SPAG-4D conversion."""
    output_path: str
    splat_count: int
    file_size: int
    processing_time: float
    depth_range: tuple


class SPAG4D:
    """
    SPAG-4D: 360° Panorama to Gaussian Splat converter.
    
    Uses PanDA or DAP for native equirectangular depth estimation
    and SPAG algorithm for Gaussian splat generation.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        use_mock_dap: bool = False,
        depth_model: str = "panda",  # "panda", "da3", "dap", or "mock"
        use_guided_filter: bool = True,
        guided_filter_radius: int = 8,
        guided_filter_eps: float = 1e-4,
        use_sharp_refinement: bool = True,
        sharp_model_path: Optional[str] = None,
        sharp_cubemap_size: int = 1536,
        sharp_projection_mode: str = "cubemap",  # "cubemap" or "icosahedral"
    ):
        """
        Initialize SPAG4D converter.

        Args:
            device: Device for computation ("cuda", "cpu", "mps")
            model_path: Optional explicit path to depth model weights
            use_mock_dap: Use mock DAP model (for testing without weights)
            depth_model: Depth model to use: "panda" (default), "da3", "dap", or "mock"
            use_guided_filter: Enable RGB-guided depth edge refinement
            guided_filter_radius: Guided filter radius (larger = smoother)
            guided_filter_eps: Guided filter regularization (smaller = sharper edges)
            use_sharp_refinement: Enable SHARP attribute refinement (default: True)
            sharp_model_path: Optional path to SHARP weights
            sharp_cubemap_size: Cubemap face size for SHARP (must be multiple of 384)
            sharp_projection_mode: "cubemap" (6 faces) or "icosahedral" (20 faces)
        """
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )

        # Handle legacy use_mock_dap parameter
        if use_mock_dap:
            depth_model = "mock"

        # Load depth model
        self.depth_model_name = depth_model
        if depth_model == "mock":
            from .dap_model import MockDAPModel
            self.dap = MockDAPModel.load(device=self.device)
        elif depth_model == "panda":
            from .panda_model import PanDAModel
            self.dap = PanDAModel.load(model_path, device=self.device)
        elif depth_model == "da3":
            from .da3_model import DA3Model
            self.dap = DA3Model.load(device=self.device)
            self.base_dap = None
            self._dap_model_path = model_path
        else:  # "dap" or fallback
            from .dap_model import DAPModel
            self.dap = DAPModel.load(model_path, device=self.device)

        # Guided depth filter (sharpens depth edges using RGB guide)
        self.guided_refiner = None
        if use_guided_filter:
            from .depth_refiner import GuidedDepthRefiner
            self.guided_refiner = GuidedDepthRefiner(
                radius=guided_filter_radius,
                eps=guided_filter_eps,
            )

        # SHARP refinement (enabled by default for maximum quality)
        self.sharp_refiner = None
        if use_sharp_refinement:
            try:
                from .sharp_refiner import SHARPRefiner
                self.sharp_refiner = SHARPRefiner(
                    device=self.device,
                    cubemap_size=sharp_cubemap_size,
                    refine_colors=True,
                    projection_mode=sharp_projection_mode,
                )
                # Preload if path provided, otherwise lazy load
                if sharp_model_path:
                    self.sharp_refiner.load_model(sharp_model_path)
            except ImportError:
                import warnings
                warnings.warn(
                    "SHARP not available. Install with: "
                    "pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip "
                    "Falling back to geometric-only Gaussians."
                )
        self.sharp_cubemap_size = sharp_cubemap_size
        self.sharp_projection_mode = sharp_projection_mode
        
        # Cache for spherical grids (by resolution + stride)
        self._grid_cache = {}
        # Lazily initialised when sharp_depth_fuse=True
        self._sharp_depth_fusion = None
    
    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        stride: int = 2,
        scale_factor: float = 1.5,
        thickness_ratio: float = 0.1,
        global_scale: float = 1.0,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        sky_threshold: float = 80.0,
        sh_degree: int = 0,
        output_format: Literal["ply", "splat"] = "ply",
        force_erp: bool = False,
        depth_preview_path: Optional[Union[str, Path]] = None,
        da3_projection: str = "equirectangular",
        # Phase 3: Sky / Pole filtering
        sky_mode: str = "sphere",          # "skip" | "sphere" | "low_opacity"
        pole_thinning: bool = True,
        sky_detection: str = "gradient",   # "gradient" | "depth" | "none"
        # Phase 2: Gaussian orientation
        oriented_gaussians: bool = True,   # depth-derived surface normals
        # Phase 5: Adaptive stride
        adaptive_stride: bool = False,
        target_gaussians: Optional[int] = None,
        edge_refine: bool = False,
        # Phase 4: Depth blending
        blend_mode: str = "laplacian",     # "feathered" | "laplacian" | "poisson"
        blend_levels: int = 5,
        # Phase 1: Cubemap SHARP Depth fusion
        sharp_depth_fuse: bool = False,
        face_size: int = 1536,
        **kwargs
    ) -> ConversionResult:
        """
        Convert equirectangular panorama to Gaussian splat.
        
        Args:
            input_path: Path to input ERP image
            output_path: Path for output file
            stride: Spatial downsampling factor (1, 2, 4, 8)
            scale_factor: Gaussian scale multiplier
            thickness_ratio: Radial thickness as fraction of tangent scales
            global_scale: Manual depth scale multiplier (for scale correction)
            depth_min: Minimum valid depth in meters
            depth_max: Maximum valid depth in meters
            sky_threshold: Depth above this is treated as sky (0 to disable)
            sh_degree: Spherical harmonics degree (0 or 3)
            output_format: Output format ("ply" or "splat")
            force_erp: Process even if aspect ratio isn't 2:1
        
        Returns:
            ConversionResult with output details
        """
        from .spherical_grid import create_spherical_grid
        from .gaussian_converter import equirect_to_gaussians, filter_sky
        from .ply_writer import save_ply_gsplat
        from .splat_writer import save_splat
        
        start_time = time.time()
        
        # Load and validate image
        img = Image.open(input_path).convert('RGB')
        img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
        
        W, H = img.size
        aspect = W / H
        
        if not (1.9 < aspect < 2.1) and not force_erp:
            raise ValueError(
                f"Image aspect ratio {aspect:.2f} is not 2:1. "
                "Use --force-erp to process anyway."
            )
            
        image_tensor = torch.from_numpy(np.array(img)).to(self.device)
        image_tensor = torch.from_numpy(np.array(img)).to(self.device)
        
        # Estimate depth with depth model (PanDA, DAP, or mock)
        with torch.inference_mode():
            if self.depth_model_name == "da3":
                if da3_projection != "equirectangular":
                    if getattr(self, "base_dap", None) is None:
                        from .dap_model import DAPModel
                        print(f"Lazy-loading DAP for DA3 {da3_projection} perspective anchoring...")
                        self.base_dap = DAPModel.load(getattr(self, "_dap_model_path", None), device=self.device)
                    depth, validity_mask = self.dap.predict(image_tensor, projection_mode=da3_projection, base_model=self.base_dap)
                else:
                    depth, validity_mask = self.dap.predict(image_tensor, projection_mode=da3_projection)
            else:
                depth, validity_mask = self.dap.predict(image_tensor)
        
        # ─────────────────────────────────────────────────────────────────
        # Phase 1: Cubemap SHARP Depth fusion (optional high-quality detail)
        # Runs SHARP on 6 cubemap faces and aligns to DAP metric space.
        # The resulting detail_depth + confidence feed into Phase 4 blending.
        # ─────────────────────────────────────────────────────────────────
        _dp_detail_np     = None   # (H, W) float32 SHARP depth, or None
        _dp_confidence_np = None   # (H, W) float32 confidence map, or None

        if sharp_depth_fuse:
            try:
                from .sharp_depth_fusion import SharpDepthFusion, ProjectionMode
                if self._sharp_depth_fusion is None or self._sharp_depth_fusion.face_size != face_size:
                    self._sharp_depth_fusion = SharpDepthFusion(
                        self.device,
                        face_size=face_size,
                        fov_deg=100.0,
                    )
                self._sharp_depth_fusion.load_model()
                _image_np_uint8 = (
                    image_tensor.cpu().numpy()
                    if image_tensor.dtype == torch.uint8
                    else (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                )
                _dap_np = depth.cpu().numpy().astype(np.float32)
                _dp_detail_np, _dp_confidence_np = self._sharp_depth_fusion.fuse(
                    _image_np_uint8, _dap_np
                )
                print(f"SHARP Depth fusion complete: face_size={face_size}")
            except Exception as e:
                import warnings
                warnings.warn(f"SHARP Depth fusion failed ({e}); using DAP depth only.")

        # Apply RGB-guided depth edge refinement
        raw_depth = depth.clone()   # keep pre-filter depth for Phase 4 blending
        if self.guided_refiner is not None:
            guided_strength = kwargs.get('guided_strength', 1.0)
            if guided_strength > 0:
                img_float = image_tensor.float() / 255.0 if image_tensor.dtype == torch.uint8 else image_tensor.float()
                depth = self.guided_refiner.refine(depth, img_float, strength=guided_strength)

        # ─────────────────────────────────────────────────────────────────
        # Phase 4: Laplacian / feathered depth blending
        # If SHARP depth ran successfully, blend DAP (low-freq) with SHARP
        # (high-freq detail) using confidence weighting.  Otherwise fall back
        # to blending raw DAP with guided-filter depth.
        # ─────────────────────────────────────────────────────────────────
        if blend_mode != "none":
            try:
                from .depth_blend import DepthBlender, BlendMode
                _blend_enum = BlendMode(blend_mode) if blend_mode != "poisson" else BlendMode.LAPLACIAN
                _blender = DepthBlender(mode=_blend_enum, n_levels=blend_levels)
                raw_np  = raw_depth.cpu().numpy().astype(np.float32)

                if _dp_detail_np is not None:
                    # Phase 1 + 4: blend DAP with SHARP using confidence mask
                    fused_np = _blender.fuse(raw_np, _dp_detail_np,
                                             confidence=_dp_confidence_np)
                    depth = torch.from_numpy(fused_np).to(self.device)
                elif self.guided_refiner is not None:
                    # Phase 4 only: blend raw DAP with guided-filtered depth
                    guided_np = depth.cpu().numpy().astype(np.float32)
                    fused_np  = _blender.fuse(raw_np, guided_np)
                    depth = torch.from_numpy(fused_np).to(self.device)
            except Exception as e:
                import warnings
                warnings.warn(f"depth_blend unavailable ({e}); skipping depth fusion.")
        
        # Save depth preview if requested
        if depth_preview_path:
            try:
                import cv2
                
                # Normalize depth for visualization (log scale for better dynamic range)
                depth_np = depth.cpu().numpy()
                depth_log = np.log1p(depth_np)
                
                # Robust min/max to avoid outliers
                d_min = np.percentile(depth_log, 1)
                d_max = np.percentile(depth_log, 99)
                
                if d_max > d_min:
                    depth_norm = np.clip((depth_log - d_min) / (d_max - d_min), 0, 1)
                    depth_norm = (depth_norm * 255).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_log, dtype=np.uint8)
                
                # Apply colormap (INFERNO is excellent for depth)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(depth_preview_path), depth_color)
            except Exception as e:
                print(f"Failed to save depth preview: {e}")
        
        # Apply global scale correction
        depth = depth * global_scale

        # ─────────────────────────────────────────────────────────────────
        # Phase 3: Scene Filtering (sky detection + pole thinning)
        # ─────────────────────────────────────────────────────────────────
        # Compute sky/pole filter masks (numpy, CPU)
        depth_np = depth.cpu().numpy().astype(np.float32)
        image_np = (
            image_tensor.cpu().numpy()
            if image_tensor.dtype != torch.uint8
            else image_tensor.cpu().numpy()
        )
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

        try:
            from .scene_filter import (
                filter_gaussian_candidates, SkyMode, apply_sky_mode_to_gaussians
            )
            _sky_mode = SkyMode(sky_mode) if sky_mode else SkyMode.BACKGROUND_SPHERE
            _keep_mask_np, _sky_mask_np = filter_gaussian_candidates(
                depth_np, image_np, stride=stride,
                sky_mode=_sky_mode,
                pole_thinning=pole_thinning,
                depth_min=depth_min,
                depth_max=depth_max,
                sky_detection=sky_detection,
            )
            # Build a full-resolution validity tensor that incorporates sky + poles
            # by upscaling the strided keep mask back to full res (nearest-neighbour)
            keep_full = torch.from_numpy(
                np.kron(_keep_mask_np, np.ones((stride, stride), dtype=np.float32))
            ).to(self.device)
            # Trim or pad to match actual image dimensions (might be off-by-one)
            keep_full = keep_full[:H, :W]
            if validity_mask is None:
                validity_mask = keep_full
            else:
                validity_mask = validity_mask * keep_full
            _use_scene_filter = True
        except Exception as e:
            import warnings
            warnings.warn(f"scene_filter unavailable ({e}); falling back to sky threshold.")
            _use_scene_filter = False
            _sky_mask_np = None
            _sky_mode = None

        if not _use_scene_filter:
            # Fallback: simple depth-threshold sky mask
            if validity_mask is None and sky_threshold > 0:
                validity_mask = (depth <= sky_threshold).float()
            elif sky_threshold > 0:
                validity_mask = validity_mask * (depth <= sky_threshold).float()
        
        # SHARP Refinement (enabled by default when refiner is available)
        refined_attrs = None
        use_sharp = kwargs.get('use_sharp_refinement', self.sharp_refiner is not None)

        if use_sharp:
            # Check if resolution changed
            current_size = self.sharp_refiner.cubemap_size if self.sharp_refiner else 0
            target_size = kwargs.get('sharp_cubemap_size', self.sharp_cubemap_size)
            
            # Re-instantiate if needed (or if not yet created)
            if self.sharp_refiner is None or current_size != target_size:
                try:
                    from .sharp_refiner import SHARPRefiner
                    if self.sharp_refiner:
                        print(f"Switching SHARP resolution: {current_size} -> {target_size}")
                    
                    self.sharp_refiner = SHARPRefiner(
                        device=self.device,
                        cubemap_size=target_size,
                        refine_colors=True,
                        projection_mode=self.sharp_projection_mode,
                    )
                    # Update default size
                    self.sharp_cubemap_size = target_size
                except ImportError:
                    use_sharp = False
            
            # Ensure model is loaded (no-op if already loaded)
            if self.sharp_refiner is not None:
                self.sharp_refiner.load_model()
            else:
                use_sharp = False
        
        # Run SHARP refinement if still enabled after all checks
        if use_sharp and self.sharp_refiner is not None:
            img_float = image_tensor.float() / 255.0
            refined_attrs = self.sharp_refiner.refine(img_float, depth)

        # ─────────────────────────────────────────────────────────────────
        # Phase 5: Adaptive stride sampling (pre-compute positions)
        # ─────────────────────────────────────────────────────────────────
        _effective_stride = stride  # used later for grid creation fallback
        if adaptive_stride or target_gaussians is not None:
            try:
                from .adaptive_stride import (
                    compute_adaptive_stride_map, refine_stride_at_edges,
                    compute_stride_for_budget
                )
                _base_stride = stride
                if target_gaussians is not None:
                    _sky_mask_for_budget = _sky_mask_np if _use_scene_filter else None
                    _base_stride = compute_stride_for_budget(
                        depth_np, target_gaussians, sky_mask=_sky_mask_for_budget
                    )
                    print(f"Target gaussians {target_gaussians:,} → computed base_stride={_base_stride}")
                if adaptive_stride:
                    _stride_map = compute_adaptive_stride_map(
                        depth_np, base_stride=_base_stride,
                        min_stride=1, max_stride=8
                    )
                    if edge_refine:
                        _stride_map = refine_stride_at_edges(_stride_map, depth_np)
                    # Use the median stride to re-create the grid at a representative resolution
                    _effective_stride = int(np.median(_stride_map))
            except Exception as e:
                import warnings
                warnings.warn(f"adaptive_stride unavailable ({e}); using uniform stride={stride}.")

        # Create the grid matched against the adaptive bounds output size
        grid_key = (H, W, _effective_stride)
        if grid_key not in self._grid_cache:
            self._grid_cache[grid_key] = create_spherical_grid(
                H, W, self.device, stride=_effective_stride
            )
        grid = self._grid_cache[grid_key]

        # Convert to Gaussians
        if refined_attrs:
            from .gaussian_converter import equirect_to_gaussians_refined
            gaussians = equirect_to_gaussians_refined(
                image=image_tensor,
                depth=depth,
                grid=grid,
                refined_attrs=refined_attrs,
                scale_factor=scale_factor,
                thickness_ratio=thickness_ratio,
                depth_min=depth_min,
                depth_max=depth_max,
                validity_mask=validity_mask,
                scale_blend=kwargs.get('scale_blend', 0.8),
                opacity_blend=kwargs.get('opacity_blend', 1.0),
                color_blend=kwargs.get('color_blend', 0.5),
                oriented_gaussians=oriented_gaussians,
            )
        else:
            gaussians = equirect_to_gaussians(
                image=image_tensor,
                depth=depth,
                grid=grid,
                scale_factor=scale_factor,
                thickness_ratio=thickness_ratio,
                depth_min=depth_min,
                depth_max=depth_max,
                validity_mask=validity_mask,
                oriented_gaussians=oriented_gaussians,
            )
        
        # ─────────────────────────────────────────────────────────────────
        # Sky handling: background sphere / low-opacity modes from scene_filter
        # ─────────────────────────────────────────────────────────────────
        if _use_scene_filter and _sky_mode is not None and _sky_mask_np is not None:
            from .scene_filter import SkyMode as _SM, apply_sky_mode_to_gaussians
            if _sky_mode != _SM.SKIP:
                # down-sample depth + image to stride resolution for sky Gaussian placement
                _depth_s  = depth_np[::stride, ::stride]
                _image_s  = image_np[::stride, ::stride]
                gaussians = apply_sky_mode_to_gaussians(
                    gaussians,
                    sky_mask_strided=_sky_mask_np,
                    depth_map_strided=_depth_s,
                    erp_image_strided=_image_s,
                    sky_mode=_sky_mode,
                    depth_max=depth_max,
                    sky_radius=kwargs.get('sky_radius', 200.0),
                    sky_opacity=0.15,
                    device=self.device,
                )
        elif kwargs.get('sky_dome', True) and sky_threshold > 0:
            # Legacy fallback sky dome
            from .gaussian_converter import generate_sky_dome
            sky_gaussians = generate_sky_dome(
                image=image_tensor,
                depth=depth,
                grid=grid,
                sky_threshold=sky_threshold,
            )
            if sky_gaussians['means'].shape[0] > 0:
                common_keys = set(gaussians.keys()) & set(sky_gaussians.keys())
                for key in common_keys:
                    gaussians[key] = torch.cat(
                        [gaussians[key], sky_gaussians[key]], dim=0
                    )
        
        # Auto-elevate sh_degree when band-1 coefficients are available
        if sh_degree == 0 and 'sh1' in gaussians:
            sh_degree = 1

        # Save output
        output_path = Path(output_path)
        if output_format == "splat":
            save_splat(gaussians, str(output_path))
        else:
            save_ply_gsplat(gaussians, str(output_path), sh_degree=sh_degree)
        
        # Compute result stats
        elapsed = time.time() - start_time
        file_size = output_path.stat().st_size
        splat_count = gaussians['means'].shape[0]
        
        # Compute actual depth range from valid points
        distances = gaussians['means'].norm(dim=-1)
        if distances.numel() > 0:
            depth_range = (distances.min().item(), distances.max().item())
        else:
            depth_range = (0.0, 0.0)
        
        return ConversionResult(
            output_path=str(output_path),
            splat_count=splat_count,
            file_size=file_size,
            processing_time=elapsed,
            depth_range=depth_range
        )
    
    def convert_ply_to_splat(self, ply_path: str, splat_path: str) -> None:
        """
        Convert existing PLY to compressed SPLAT format.
        
        Args:
            ply_path: Input PLY file path
            splat_path: Output SPLAT file path
        """
        from .splat_writer import convert_ply_to_splat
        convert_ply_to_splat(ply_path, splat_path)
    
    def clear_cache(self) -> None:
        """Clear cached spherical grids to free memory."""
        self._grid_cache.clear()
