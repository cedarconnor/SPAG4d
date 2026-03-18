# spag4d/core.py
"""
Main SPAG4D class that orchestrates the conversion pipeline.

Two modes:
  - SPAG (default): DAP/DA360 depth -> direct spherical Gaussian conversion
  - SHARP refined:  DAP/DA360 depth -> per-face SHARP inference -> merged Gaussians
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
import time

from PIL import Image, ImageOps


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

    Supports two pipelines:
      - SPAG (default): Fast depth-to-Gaussian conversion via spherical projection
      - SHARP refined: Per-face SHARP inference for higher quality output
    """

    def __init__(
        self,
        device: str = "cuda",
        depth_model: str = "da360",
        model_path: Optional[str] = None,
        use_mock_dap: bool = False,
        sharp_refine: bool = False,
        sharp_cubemap_size: int = 1536,
        sharp_projection_mode: str = "icosahedral",
    ):
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )

        self.default_depth_model = depth_model
        self.use_mock_dap = use_mock_dap
        self._depth_models = {}  # lazy-loaded cache: name -> model

        # Eagerly load the default depth model
        self._get_depth_model(depth_model)

        self.sharp_refine = sharp_refine
        self.sharp_cubemap_size = sharp_cubemap_size
        self.sharp_projection_mode = sharp_projection_mode
        self._sharp_pipeline = None

    def _get_depth_model(self, name: str):
        """Get or lazy-load a depth model by name."""
        if name in self._depth_models:
            return self._depth_models[name]

        if self.use_mock_dap:
            from .dap_model import MockDAPModel
            model = MockDAPModel.load(device=self.device)
        elif name == "da360":
            from .da360_model import DA360Model
            model = DA360Model.load(device=self.device)
        else:
            from .dap_model import DAPModel
            model = DAPModel.load(device=self.device)

        self._depth_models[name] = model
        return model

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        sky_threshold: float = 80.0,
        stride: int = 2,
        outlier_pruning: float = 0.0,
        global_scale: float = 1.0,
        force_erp: bool = False,
        depth_model: Optional[str] = None,
        sharp_refine: Optional[bool] = None,
        sharp_projection: Optional[str] = None,
        sharp_cubemap_size: Optional[int] = None,
        grid_jitter: float = 0.0,
        depth_preview_path: Optional[Union[str, Path]] = None,
    ) -> ConversionResult:
        """
        Convert equirectangular panorama to Gaussian splat PLY.

        Args:
            input_path: Path to input ERP image
            output_path: Path for output PLY file
            depth_min: Minimum valid depth in meters
            depth_max: Maximum valid depth in meters
            sky_threshold: Depth above this is treated as sky (0 to disable)
            stride: SPAG pixel stride (1=full density, 2=quarter, 4=sixteenth)
            outlier_pruning: Statistical outlier removal strength (0=off, 1=aggressive)
            global_scale: Manual depth scale multiplier
            force_erp: Process even if aspect ratio isn't 2:1
            depth_model: Override depth model ("dap" or "da360")
            sharp_refine: Override instance-level SHARP refinement setting
            sharp_projection: Override projection mode ("cubemap" or "icosahedral")
            sharp_cubemap_size: Override cubemap face size
            grid_jitter: Grid jitter for SHARP path (0=off)
            depth_preview_path: Optional path to save depth visualization

        Returns:
            ConversionResult with output details
        """
        from .ply_writer import save_ply_gsplat

        start_time = time.time()

        # Load and validate image
        img = Image.open(input_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        W, H = img.size
        aspect = W / H

        if not (1.9 < aspect < 2.1) and not force_erp:
            raise ValueError(
                f"Image aspect ratio {aspect:.2f} is not 2:1. "
                "Use --force-erp to process anyway."
            )

        image_tensor = torch.from_numpy(np.array(img)).to(self.device)

        # Estimate depth
        dm_name = depth_model or self.default_depth_model
        depth_engine = self._get_depth_model(dm_name)
        print(f"[SPAG4D] Running {dm_name.upper()} depth estimation...", flush=True)
        t_depth = time.time()
        with torch.inference_mode():
            depth_raw, _ = depth_engine.predict(image_tensor)
        depth = depth_raw * global_scale
        print(f"[SPAG4D] Depth estimation complete in {time.time() - t_depth:.1f}s")

        # Save depth preview if requested
        if depth_preview_path:
            self._save_depth_preview(depth, depth_preview_path)

        # Decide pipeline
        use_sharp = sharp_refine if sharp_refine is not None else self.sharp_refine

        if use_sharp:
            gaussians = self._run_sharp_pipeline(
                image_tensor, depth,
                depth_min=depth_min, depth_max=depth_max,
                sky_threshold=sky_threshold, grid_jitter=grid_jitter,
                sharp_projection=sharp_projection,
                sharp_cubemap_size=sharp_cubemap_size,
            )
            colors_linear = True
        else:
            gaussians = self._run_spag_pipeline(
                image_tensor, depth,
                depth_min=depth_min, depth_max=depth_max,
                sky_threshold=sky_threshold, stride=stride,
            )
            colors_linear = False

        # Outlier pruning
        if outlier_pruning > 0.0 and gaussians['means'].shape[0] > 0:
            try:
                from .scene_filter import prune_outliers
                gaussians = prune_outliers(gaussians, strength=outlier_pruning)
            except Exception as e:
                import warnings
                warnings.warn(f"Outlier pruning failed: {e}")

        # Save PLY
        output_path = str(output_path)
        n_gaussians = gaussians['means'].shape[0]
        print(f"[SPAG4D] Saving PLY ({n_gaussians:,} Gaussians)...", flush=True)
        t_save = time.time()
        save_ply_gsplat(gaussians, output_path, sh_degree=0, colors_linear=colors_linear)
        print(f"[SPAG4D] Save complete in {time.time() - t_save:.1f}s")

        processing_time = time.time() - start_time
        file_size = Path(output_path).stat().st_size

        # Compute actual depth range
        distances = gaussians['means'].norm(dim=-1)
        if distances.numel() > 0:
            depth_range = (distances.min().item(), distances.max().item())
        else:
            depth_range = (0.0, 0.0)

        return ConversionResult(
            output_path=output_path,
            splat_count=n_gaussians,
            file_size=file_size,
            processing_time=processing_time,
            depth_range=depth_range,
        )

    def _run_spag_pipeline(
        self,
        image_tensor: torch.Tensor,
        depth: torch.Tensor,
        depth_min: float,
        depth_max: float,
        sky_threshold: float,
        stride: int,
    ) -> dict:
        """SPAG path: direct depth-to-Gaussian conversion."""
        from .spag_converter import depth_to_gaussians, SPAGParams

        print(f"[SPAG4D] SPAG conversion (stride={stride})...", flush=True)
        t = time.time()

        params = SPAGParams(
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            sky_detection="depth",
        )

        gaussians = depth_to_gaussians(
            erp_image=image_tensor,
            depth_map=depth,
            params=params,
            device=self.device,
        )

        n = gaussians['means'].shape[0]
        print(f"[SPAG4D] SPAG generated {n:,} Gaussians in {time.time() - t:.1f}s")
        return gaussians

    def _run_sharp_pipeline(
        self,
        image_tensor: torch.Tensor,
        depth: torch.Tensor,
        depth_min: float,
        depth_max: float,
        sky_threshold: float,
        grid_jitter: float,
        sharp_projection: Optional[str],
        sharp_cubemap_size: Optional[int],
    ) -> dict:
        """SHARP refinement path: per-face SHARP inference."""
        from .sharp_gaussian_pipeline import SHARPGaussianPipeline

        cubemap_sz = sharp_cubemap_size or self.sharp_cubemap_size
        proj_mode = sharp_projection or self.sharp_projection_mode

        # Lazy init / rebuild if settings changed
        if (self._sharp_pipeline is not None and
            (self._sharp_pipeline.projection_mode != proj_mode or
             self._sharp_pipeline.cubemap_size != cubemap_sz)):
            self._sharp_pipeline = None

        if self._sharp_pipeline is None:
            self._sharp_pipeline = SHARPGaussianPipeline(
                device=self.device,
                cubemap_size=cubemap_sz,
                projection_mode=proj_mode,
            )
            self._sharp_pipeline.load_model()

        erp_float = image_tensor.float() / 255.0 if image_tensor.dtype == torch.uint8 else image_tensor.float()
        gaussians = self._sharp_pipeline.generate(
            erp_image=erp_float,
            dap_depth=depth,
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            grid_jitter=grid_jitter,
        )

        return gaussians

    @staticmethod
    def _save_depth_preview(depth: torch.Tensor, path: Union[str, Path]):
        """Save depth map visualization as JPEG."""
        try:
            import cv2

            depth_np = depth.cpu().numpy()
            depth_log = np.log1p(depth_np)

            d_min = np.percentile(depth_log, 1)
            d_max = np.percentile(depth_log, 99)

            if d_max > d_min:
                depth_norm = np.clip((depth_log - d_min) / (d_max - d_min), 0, 1)
                depth_norm = (depth_norm * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_log, dtype=np.uint8)

            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(path), depth_color)
        except Exception as e:
            print(f"Failed to save depth preview: {e}")
