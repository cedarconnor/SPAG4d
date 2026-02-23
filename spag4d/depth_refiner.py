# spag4d/depth_refiner.py
"""
RGB-guided depth edge refinement.

Uses guided filtering to transfer sharp edges from the RGB panorama
onto the depth map, producing depth maps with crisp boundaries while
preserving global depth structure.

Based on "Guided Image Filtering" (He et al., 2010/2013).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GuidedDepthRefiner:
    """
    Refine depth edges using the panorama RGB image as a guide.

    The guided filter uses RGB edges to sharpen corresponding depth edges
    without altering the global depth structure. This is particularly
    effective for 360° depth maps where model outputs tend to have
    blurry object boundaries.

    Pipeline:
        1. Convert depth + RGB to numpy
        2. Apply guided filter (RGB guides depth edge placement)
        3. Return sharpened depth tensor
    """

    def __init__(
        self,
        radius: int = 8,
        eps: float = 1e-4,
        use_opencv: bool = True,
    ):
        """
        Args:
            radius: Filter radius (larger = smoother, but preserves more edges)
            eps: Regularization (smaller = sharper edges, risk of artifacts)
            use_opencv: Try OpenCV ximgproc first (faster), fallback to pure Python
        """
        self.radius = radius
        self.eps = eps
        self.use_opencv = use_opencv
        self._has_ximgproc = None  # Lazy check

    def refine(
        self,
        depth: torch.Tensor,
        rgb_guide: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Refine depth edges using RGB guidance.

        Args:
            depth: Depth map [H, W] (any scale/range)
            rgb_guide: RGB image [H, W, 3] float [0, 1]
            strength: Blend factor (0.0 = original, 1.0 = fully refined)

        Returns:
            Refined depth [H, W] with sharper edges
        """
        device = depth.device

        # validation
        strength = max(0.0, min(1.0, strength))
        if strength <= 0.001:
            return depth

        # Convert to numpy for filtering
        depth_np = depth.cpu().numpy().astype(np.float32)
        guide_np = rgb_guide.cpu().numpy().astype(np.float32)

        # Try OpenCV path first (faster), fall back to pure Python
        result_np = self._try_opencv_guided_filter(depth_np, guide_np)
        if result_np is None:
            result_np = self._guided_filter_python(depth_np, guide_np)

        # Convert back to tensor
        result = torch.from_numpy(result_np).to(device)

        # Do NOT globally rescale result to match original min/max — that would
        # undo the local edge sharpening that guided filtering was supposed to
        # provide (sharpening one edge forces every other depth to shift).
        # The strength blend below is sufficient for range preservation.

        # Apply strength blending
        if strength < 1.0:
            result = result * strength + depth * (1.0 - strength)

        # Prevent guided filter ringing overshoots from pushing depth < 0
        result = result.clamp(min=1e-4)

        return result


    def _try_opencv_guided_filter(
        self,
        depth_np: np.ndarray,
        guide_np: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Try OpenCV ximgproc guided filter. Returns None if unavailable."""
        if not self.use_opencv:
            return None

        if self._has_ximgproc is None:
            try:
                import cv2
                # Check if ximgproc is available
                cv2.ximgproc.guidedFilter
                self._has_ximgproc = True
            except (ImportError, AttributeError):
                self._has_ximgproc = False
                print("[GuidedFilter] OpenCV ximgproc not available, using pure Python fallback")

        if not self._has_ximgproc:
            return None

        import cv2

        # OpenCV guidedFilter expects guide as uint8 or float32
        guide_uint8 = (guide_np * 255).clip(0, 255).astype(np.uint8)

        # Apply guided filter
        result = cv2.ximgproc.guidedFilter(
            guide=guide_uint8,
            src=depth_np,
            radius=self.radius,
            eps=self.eps,
        )

        return result

    def _guided_filter_python(
        self,
        src: np.ndarray,
        guide: np.ndarray,
    ) -> np.ndarray:
        """
        Pure Python/NumPy guided filter implementation.

        Based on the O(N) box-filter formulation from He et al. 2013.
        """
        r = self.radius
        eps = self.eps

        # Convert guide to grayscale if color
        if guide.ndim == 3:
            guide_gray = np.mean(guide, axis=2).astype(np.float32)
        else:
            guide_gray = guide.astype(np.float32)

        src = src.astype(np.float32)

        # Box filter using cumulative sum (O(N) per channel)
        def box_filter(img, r):
            """Fast box filter using integral image."""
            H, W = img.shape[:2]
            # Pad
            padded = np.pad(img, ((r, r), (r, r)), mode='reflect')
            # Integral image
            integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            # Box filter result
            result = (
                integral[2*r:, 2*r:]
                - integral[:H, 2*r:]
                - integral[2*r:, :W]
                + integral[:H, :W]
            )
            # Normalize by window size
            count = (2 * r + 1) ** 2
            return result / count

        # Guided filter core
        mean_I = box_filter(guide_gray, r)
        mean_p = box_filter(src, r)
        mean_Ip = box_filter(guide_gray * src, r)
        mean_II = box_filter(guide_gray * guide_gray, r)

        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)

        result = mean_a * guide_gray + mean_b

        return result
