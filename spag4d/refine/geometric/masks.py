# spag4d/refine/geometric/masks.py
"""Sky, specular, and composite validity masks for depth alignment."""
import cv2
import numpy as np
from scipy.ndimage import binary_erosion


def sky_mask_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask of likely sky pixels using HSV heuristics.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
    Returns:
        (H, W) bool mask, True = sky.
    """
    H = image_rgb.shape[0]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0      # brightness
    saturation = hsv[:, :, 1] / 255.0

    # Sky: high value, low saturation
    bright = value > 0.75
    low_sat = saturation < 0.25

    # Weight toward upper hemisphere of ERP
    row_weight = np.linspace(1.0, 0.0, H)[:, np.newaxis]
    upper_half = row_weight > 0.5

    return (bright & low_sat & upper_half).astype(bool)


def specular_mask(image_rgb: np.ndarray, window: int = 9) -> np.ndarray:
    """Return boolean mask of likely specular/blown-out pixels.

    Specular: high luminance + low local texture variance.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        window: local variance window size.
    Returns:
        (H, W) bool mask, True = specular.
    """
    gray = image_rgb.mean(axis=2).astype(np.float32) / 255.0
    high_lum = gray > 0.85

    # Local texture variance via blur difference
    blur = cv2.GaussianBlur(gray, (window, window), 0)
    var = np.abs(gray - blur)
    low_var = var < 0.02

    return (high_lum & low_var).astype(bool)


def build_validity_mask(
    alpha: np.ndarray,
    depth: np.ndarray,
    sky_mask: np.ndarray,
    specular_mask_arr: np.ndarray,
    alpha_confident: float = 0.90,
    far_clamp_ratio: float = 0.95,
    far: float = 1000.0,
    gradient_threshold: float = 0.15,
    erode_px: int = 3,
) -> np.ndarray:
    """Build composite validity mask for depth alignment.

    Excludes: low alpha, far-plane, high-gradient, sky, specular, plus erosion.
    Returns (H, W) bool mask, True = valid for alignment.
    """
    alpha_ok = alpha >= alpha_confident
    depth_ok = depth < (far_clamp_ratio * far)
    not_sky = ~sky_mask
    not_spec = ~specular_mask_arr

    # Depth gradient mask
    gy, gx = np.gradient(depth)
    grad_rel = np.sqrt(gx**2 + gy**2) / (np.abs(depth) + 1e-6)
    low_grad = grad_rel < gradient_threshold

    combined = alpha_ok & depth_ok & not_sky & not_spec & low_grad

    # Erode to avoid boundary bleed
    if erode_px > 0:
        struct = np.ones((erode_px, erode_px), dtype=bool)
        combined = binary_erosion(combined, structure=struct)

    return combined.astype(bool)
