"""SHARP 360 pipeline: per-face SHARP prediction with DA360 alignment.

Orchestrates the full conversion of a 360 equirectangular panorama into a
merged 3D Gaussian Splat via:
  1. Perspective face extraction from ERP panorama
  2. Optional SeedVR2 upscaling
  3. DA360 disparity prediction on full panorama
  4. Per-face SHARP Gaussian prediction
  5. Hard Voronoi border clipping
  6. DA360 grid-based depth alignment
  7. World-frame rotation and merge
  8. PLY export via SHARP's save_ply()
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# SHARP vendored source path setup
_SHARP_SRC = str(Path(__file__).resolve().parent / "sharp_arch" / "ml-sharp" / "src")
if _SHARP_SRC not in sys.path:
    sys.path.insert(0, _SHARP_SRC)

LOGGER = logging.getLogger(__name__)

PI = math.pi
TWO_PI = 2.0 * PI

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceOrientation:
    """Orientation of a single perspective view extracted from an ERP panorama.

    Axes follow right-hand convention:
      - right:   positive-X of the view
      - down:    positive-Y of the view
      - forward: positive-Z (into the scene)
    """

    name: str
    right: np.ndarray    # (3,)
    down: np.ndarray     # (3,)
    forward: np.ndarray  # (3,)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation from view-local to world coordinates.

        Columns are the view axes expressed in the world frame:
          R = [right | down | forward]

        So ``world_point = R @ view_point``.
        """
        return np.column_stack([self.right, self.down, self.forward])


@dataclass(frozen=True)
class ExtractionLayout:
    """Describes a set of perspective views to extract from an ERP panorama."""

    name: str
    views: List[FaceOrientation]
    focal_px: float      # horizontal focal length in pixels
    focal_y_px: float    # vertical focal length in pixels
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# View geometry helpers
# ---------------------------------------------------------------------------

def make_horizon_view(index: int, side_count: int) -> FaceOrientation:
    """Create a single horizon-ring FaceOrientation at the given index.

    The view faces outward along the horizon at azimuth
    ``index * (360 / side_count)`` degrees, measured from +Z towards +X.

    Naming convention:
      - 2 sides: "front", "back"
      - 4 sides: "front", "right", "back", "left"
      - otherwise: "side_01", "side_02", ...
    """
    azimuth_rad = index * TWO_PI / side_count

    # Forward direction on the horizon (Y = 0)
    forward = np.array([
        math.sin(azimuth_rad),
        0.0,
        math.cos(azimuth_rad),
    ], dtype=np.float64)

    # Camera Y-axis points downward in image space.
    # SHARP convention: down = +Y (matching Apple's ml-sharp and SHARP_360_to_Splat)
    down = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Right vector — explicit formula matching SHARP_360_to_Splat reference
    right = np.array([
        math.cos(azimuth_rad),
        0.0,
        -math.sin(azimuth_rad),
    ], dtype=np.float64)

    # Name
    if side_count == 2:
        names = ["front", "back"]
        name = names[index]
    elif side_count == 4:
        names = ["front", "right", "back", "left"]
        name = names[index]
    else:
        name = f"side_{index + 1:02d}"

    return FaceOrientation(name=name, right=right, down=down, forward=forward)


def build_extraction_layout(
    face_size: int,
    panorama_height: int,
    side_count: int = 6,
    overlap_degrees: float = 10.0,
) -> ExtractionLayout:
    """Build an extraction layout for *side_count* horizon views.

    Matches the SHARP_360_to_Splat reference:
    - image_width is widened to accommodate the overlap FOV
    - image_height = panorama_height (tall faces for full vertical coverage)
    - focal_y = focal_x (square pixels)

    Args:
        face_size: Base face width in pixels (before overlap widening).
        panorama_height: Height of the input ERP panorama in pixels.
        side_count: Number of horizon views (default 6).
        overlap_degrees: Extra FOV overlap per side in degrees (default 10).

    Returns:
        An ExtractionLayout with the computed views and intrinsics.
    """
    span_deg = 360.0 / side_count
    view_fov_deg = min(170.0, span_deg + overlap_degrees)

    # Widen image to accommodate the overlap FOV (reference: insp_to_splat.py:1223)
    image_width = max(face_size, int(round(face_size * (view_fov_deg / span_deg))))

    # Focal length from the widened image and total FOV
    focal_px = (image_width / 2.0) / math.tan(math.radians(view_fov_deg) / 2.0)

    # Tall faces: image_height = panorama_height for full vertical coverage
    # (reference: insp_to_splat.py:1225 with cutoff_height_percent=0)
    image_height = panorama_height

    # Square pixels: focal_y = focal_x
    focal_y_px = focal_px

    views = [make_horizon_view(i, side_count) for i in range(side_count)]

    name = f"horizon_{side_count}_{face_size}px"
    return ExtractionLayout(
        name=name,
        views=views,
        focal_px=focal_px,
        focal_y_px=focal_y_px,
        image_width=image_width,
        image_height=image_height,
    )


# ---------------------------------------------------------------------------
# Bilinear sampling with ERP horizontal wrap
# ---------------------------------------------------------------------------

def bilinear_sample(
    image: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
) -> np.ndarray:
    """Bilinear-sample an RGB image with horizontal wrap.

    Args:
        image: (H, W, C) uint8 or float array.
        sample_x: (...) float pixel x coordinates (0-based).
        sample_y: (...) float pixel y coordinates (0-based).

    Returns:
        (..., C) sampled values, same dtype as input.
    """
    H, W = image.shape[:2]
    C = image.shape[2] if image.ndim == 3 else 1
    img = image.astype(np.float64).reshape(H, W, C)

    x0 = np.floor(sample_x).astype(np.int64)
    y0 = np.floor(sample_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    fx = (sample_x - x0).astype(np.float64)
    fy = (sample_y - y0).astype(np.float64)

    # Horizontal wrap, vertical clamp
    x0w = x0 % W
    x1w = x1 % W
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Four corners
    v00 = img[y0c, x0w]
    v10 = img[y0c, x1w]
    v01 = img[y1c, x0w]
    v11 = img[y1c, x1w]

    # Bilinear weights
    w00 = ((1.0 - fx) * (1.0 - fy))[..., None]
    w10 = (fx * (1.0 - fy))[..., None]
    w01 = ((1.0 - fx) * fy)[..., None]
    w11 = (fx * fy)[..., None]

    result = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11

    if image.dtype == np.uint8:
        result = np.clip(result + 0.5, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        result = result[..., 0]

    return result


def bilinear_sample_scalar(
    image: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
) -> np.ndarray:
    """Bilinear-sample a scalar (2-D) map with horizontal wrap.

    Args:
        image: (H, W) float array.
        sample_x: (...) float pixel x coordinates.
        sample_y: (...) float pixel y coordinates.

    Returns:
        (...) sampled values.
    """
    H, W = image.shape[:2]

    x0 = np.floor(sample_x).astype(np.int64)
    y0 = np.floor(sample_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (sample_x - x0).astype(np.float64)
    fy = (sample_y - y0).astype(np.float64)

    x0w = x0 % W
    x1w = x1 % W
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    v00 = image[y0c, x0w].astype(np.float64)
    v10 = image[y0c, x1w].astype(np.float64)
    v01 = image[y1c, x0w].astype(np.float64)
    v11 = image[y1c, x1w].astype(np.float64)

    return (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
    )


# ---------------------------------------------------------------------------
# Perspective extraction from ERP
# ---------------------------------------------------------------------------

def _pixel_ray_directions(
    image_width: int,
    image_height: int,
    focal_x_px: float,
    focal_y_px: float,
    view: FaceOrientation,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute world-frame ray directions for every pixel in a perspective view.

    Returns (world_x, world_y, world_z) each of shape (image_height, image_width).
    """
    # Pixel grid — centre of each pixel
    u = np.arange(image_width, dtype=np.float64) + 0.5
    v = np.arange(image_height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # View-local ray (pinhole camera, origin at image centre)
    cx = image_width / 2.0
    cy = image_height / 2.0
    local_x = (uu - cx) / focal_x_px
    local_y = (vv - cy) / focal_y_px
    local_z = np.ones_like(uu)

    # Rotate to world frame
    R = view.rotation_matrix  # 3x3, columns = [right, down, forward]
    world_x = R[0, 0] * local_x + R[0, 1] * local_y + R[0, 2] * local_z
    world_y = R[1, 0] * local_x + R[1, 1] * local_y + R[1, 2] * local_z
    world_z = R[2, 0] * local_x + R[2, 1] * local_y + R[2, 2] * local_z

    return world_x, world_y, world_z


def _world_to_erp_pixels(
    world_x: np.ndarray,
    world_y: np.ndarray,
    world_z: np.ndarray,
    erp_width: int,
    erp_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert world-frame direction vectors to ERP pixel coordinates.

    Uses the convention from SHARP_360_to_Splat:
      longitude = atan2(x, z)
      latitude  = asin(y / ||dir||)
    """
    norm = np.sqrt(world_x ** 2 + world_y ** 2 + world_z ** 2)
    norm = np.maximum(norm, 1e-12)

    longitude = np.arctan2(world_x, world_z)                  # (-pi, pi]
    latitude = np.arcsin(np.clip(world_y / norm, -1.0, 1.0))  # (-pi/2, pi/2)

    sample_x = (longitude / TWO_PI + 0.5) * erp_width - 0.5
    sample_y = (latitude / PI + 0.5) * erp_height - 0.5

    return sample_x, sample_y


def extract_perspective_view(
    panorama: np.ndarray,
    image_width: int,
    image_height: int,
    focal_x_px: float,
    focal_y_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    """Extract a single perspective face from an ERP panorama.

    Args:
        panorama: (H, W, 3) uint8 ERP image.
        image_width: Output face width in pixels.
        image_height: Output face height in pixels.
        focal_x_px: Horizontal focal length in pixels.
        focal_y_px: Vertical focal length in pixels.
        view: FaceOrientation describing the view direction.

    Returns:
        (image_height, image_width, 3) uint8 face image.
    """
    erp_h, erp_w = panorama.shape[:2]
    wx, wy, wz = _pixel_ray_directions(image_width, image_height, focal_x_px, focal_y_px, view)
    sx, sy = _world_to_erp_pixels(wx, wy, wz, erp_w, erp_h)
    return bilinear_sample(panorama, sx, sy)


def extract_perspective_views(
    layout: ExtractionLayout,
    panorama: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract all perspective faces from an ERP panorama.

    Returns:
        Dict mapping view name to (H, W, 3) uint8 face image.
    """
    faces: Dict[str, np.ndarray] = {}
    for view in layout.views:
        face = extract_perspective_view(
            panorama,
            layout.image_width,
            layout.image_height,
            layout.focal_px,
            layout.focal_y_px,
            view,
        )
        faces[view.name] = face
    return faces


def extract_perspective_scalar_view(
    scalar_map: np.ndarray,
    image_width: int,
    image_height: int,
    focal_x_px: float,
    focal_y_px: float,
    view: FaceOrientation,
) -> np.ndarray:
    """Extract a single perspective scalar map from an ERP scalar map.

    Args:
        scalar_map: (H, W) float scalar ERP map (e.g. disparity).
        image_width: Output width.
        image_height: Output height.
        focal_x_px: Horizontal focal length.
        focal_y_px: Vertical focal length.
        view: FaceOrientation.

    Returns:
        (image_height, image_width) float scalar face.
    """
    erp_h, erp_w = scalar_map.shape[:2]
    wx, wy, wz = _pixel_ray_directions(image_width, image_height, focal_x_px, focal_y_px, view)
    sx, sy = _world_to_erp_pixels(wx, wy, wz, erp_w, erp_h)
    return bilinear_sample_scalar(scalar_map, sx, sy)


# ---------------------------------------------------------------------------
# Gaussian operations
# ---------------------------------------------------------------------------

def filter_gaussians_by_view_border(
    gaussians: "Gaussians3D",
    horizontal_degrees: float,
    vertical_degrees: float = None,
) -> "Gaussians3D":
    """Hard Voronoi clipping: keep Gaussians within a horizontal angular strip.

    Clips **horizontally only** by default (matching SHARP_360_to_Splat
    reference), preserving full vertical extent. Uses the tan-ratio in
    image-plane coordinates, not a spherical cone.

    Args:
        gaussians: Gaussians3D with tensors of shape (1, N, D).
        horizontal_degrees: Total horizontal clip width in degrees.
        vertical_degrees: Optional total vertical clip height in degrees.
            If None, no vertical clipping is applied.

    Returns:
        Filtered Gaussians3D.
    """
    from sharp.utils.gaussians import Gaussians3D as _G3D

    means = gaussians.mean_vectors[0]  # (N, 3)
    depth = means[:, 2]  # view-local Z

    # Horizontal limit: |X/Z| <= tan(half_horizontal)
    if horizontal_degrees >= 179.0:
        h_limit = float("inf")
    else:
        h_limit = math.tan(math.radians(horizontal_degrees / 2.0))

    h_ratio = torch.abs(means[:, 0]) / depth.clamp(min=1e-6)
    mask = (depth > 0.0) & (h_ratio <= h_limit)

    # Optional vertical clipping
    if vertical_degrees is not None and vertical_degrees > 0.0:
        if vertical_degrees >= 179.0:
            v_limit = float("inf")
        else:
            v_limit = math.tan(math.radians(vertical_degrees / 2.0))
        v_ratio = torch.abs(means[:, 1]) / depth.clamp(min=1e-6)
        mask = mask & (v_ratio <= v_limit)

    return _G3D(
        mean_vectors=gaussians.mean_vectors[:, mask],
        singular_values=gaussians.singular_values[:, mask],
        quaternions=gaussians.quaternions[:, mask],
        colors=gaussians.colors[:, mask],
        opacities=gaussians.opacities[:, mask],
    )


def align_gaussians_to_reference(
    gaussians: "Gaussians3D",
    ref_disparity_view: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    image_width: int,
    image_height: int,
    grid_resolution: int = 8,
) -> "Gaussians3D":
    """Align Gaussian depths to a DA360 reference disparity via a smooth
    NxN grid scale field.

    The Gaussians are assumed to be in view-local coordinates where the depth
    axis is +Z.  The reference disparity is the DA360 output projected into the
    same view.

    Args:
        gaussians: Gaussians3D (1, N, D) in view-local coordinates.
        ref_disparity_view: (H, W) float disparity map for this view.
        focal_x_px: Horizontal focal length.
        focal_y_px: Vertical focal length.
        image_width: View width in pixels.
        image_height: View height in pixels.
        grid_resolution: NxN grid cells for the smooth scale field.

    Returns:
        Depth-aligned Gaussians3D.
    """
    from sharp.utils.gaussians import Gaussians3D as _G3D

    device = gaussians.mean_vectors.device
    mv = gaussians.mean_vectors  # (1, N, 3)
    mv_np = mv[0].detach().cpu().numpy().astype(np.float32)
    N = mv_np.shape[0]

    if N == 0:
        return gaussians

    depth_z = mv_np[:, 2]
    radial = np.linalg.norm(mv_np, axis=1)

    # Project Gaussian centres to pixel coordinates
    valid = depth_z > 1e-6
    px_x = (mv_np[:, 0] / np.clip(depth_z, 1e-6, None)) * focal_x_px + (image_width / 2.0) - 0.5
    px_y = (mv_np[:, 1] / np.clip(depth_z, 1e-6, None)) * focal_y_px + (image_height / 2.0) - 0.5
    valid &= (px_x >= 0) & (px_x <= image_width - 1)
    valid &= (px_y >= 0) & (px_y <= image_height - 1)

    # Aspect-ratio-aware grid (reference: insp_to_splat.py:2057)
    grid_cx = max(1, int(grid_resolution))
    grid_cy = max(1, int(round(grid_cx * (image_height / max(1, image_width)))))

    per_point_scale = np.ones(N, dtype=np.float32)
    median_scale = 1.0
    count = 0

    if int(valid.sum()) >= 64:
        # Sample reference disparity at valid Gaussian pixel locations
        ref_disp = bilinear_sample_scalar(
            ref_disparity_view, px_x[valid], px_y[valid],
        )
        ok = np.isfinite(ref_disp) & (ref_disp > 1e-6) & (radial[valid] > 1e-6)
        count = int(ok.sum())

        if count >= 64:
            # Compute scale ratio in depth space: ref_depth / sharp_radial
            ref_depth_ok = (1.0 / ref_disp[ok]).astype(np.float32)
            sharp_r_ok = radial[valid][ok]
            raw_scale = ref_depth_ok / sharp_r_ok

            # Global robust median
            lo, hi = np.quantile(raw_scale, [0.05, 0.95])
            trimmed = raw_scale[(raw_scale >= lo) & (raw_scale <= hi)]
            median_scale = (
                float(np.median(trimmed)) if trimmed.size > 0
                else float(np.median(raw_scale))
            )

            # Build coarse NxM grid of median scales
            px_ok = px_x[valid][ok]
            py_ok = px_y[valid][ok]
            cell_w = image_width / grid_cx
            cell_h = image_height / grid_cy
            grid = np.full((grid_cy, grid_cx), median_scale, dtype=np.float32)

            for gy in range(grid_cy):
                for gx_idx in range(grid_cx):
                    in_cell = (
                        (px_ok >= gx_idx * cell_w) & (px_ok < (gx_idx + 1) * cell_w) &
                        (py_ok >= gy * cell_h) & (py_ok < (gy + 1) * cell_h)
                    )
                    if int(in_cell.sum()) >= 8:
                        cs = raw_scale[in_cell]
                        cl, ch = np.quantile(cs, [0.1, 0.9])
                        ct = cs[(cs >= cl) & (cs <= ch)]
                        if ct.size > 0:
                            grid[gy, gx_idx] = float(np.median(ct))

            grid = np.clip(grid, median_scale * 0.1, median_scale * 10.0)

            # Bilinear interpolate grid to every Gaussian
            all_px = np.clip(px_x, 0, image_width - 1)
            all_py = np.clip(px_y, 0, image_height - 1)
            gxc = all_px / cell_w - 0.5
            gyc = all_py / cell_h - 0.5
            gx0 = np.clip(np.floor(gxc).astype(np.int32), 0, grid_cx - 1)
            gy0 = np.clip(np.floor(gyc).astype(np.int32), 0, grid_cy - 1)
            gx1 = np.clip(gx0 + 1, 0, grid_cx - 1)
            gy1 = np.clip(gy0 + 1, 0, grid_cy - 1)
            wx = np.clip(gxc - gx0, 0, 1).astype(np.float32)
            wy = np.clip(gyc - gy0, 0, 1).astype(np.float32)
            per_point_scale = (
                grid[gy0, gx0] * (1 - wx) * (1 - wy) +
                grid[gy0, gx1] * wx * (1 - wy) +
                grid[gy1, gx0] * (1 - wx) * wy +
                grid[gy1, gx1] * wx * wy
            )
            per_point_scale[~valid] = median_scale

    scale_t = (
        torch.from_numpy(per_point_scale)
        .to(device=device, dtype=mv.dtype)
        .unsqueeze(0).unsqueeze(-1)
    )  # (1, N, 1)

    return _G3D(
        mean_vectors=mv * scale_t,
        singular_values=gaussians.singular_values * scale_t,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    ), median_scale, count


def scale_gaussians(
    gaussians: "Gaussians3D",
    scale_factor: float,
) -> "Gaussians3D":
    """Uniformly scale Gaussian positions and singular values."""
    from sharp.utils.gaussians import Gaussians3D as _G3D

    return _G3D(
        mean_vectors=gaussians.mean_vectors * scale_factor,
        singular_values=gaussians.singular_values * scale_factor,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def merge_gaussians(
    gaussians_list: List["Gaussians3D"],
) -> "Gaussians3D":
    """Concatenate a list of Gaussians3D into one (along N dimension).

    All inputs must have batch dim 1, i.e. shape (1, N_i, D).
    """
    from sharp.utils.gaussians import Gaussians3D as _G3D

    return _G3D(
        mean_vectors=torch.cat([g.mean_vectors for g in gaussians_list], dim=1),
        singular_values=torch.cat([g.singular_values for g in gaussians_list], dim=1),
        quaternions=torch.cat([g.quaternions for g in gaussians_list], dim=1),
        colors=torch.cat([g.colors for g in gaussians_list], dim=1),
        opacities=torch.cat([g.opacities for g in gaussians_list], dim=1),
    )


# ---------------------------------------------------------------------------
# DA360 integration
# ---------------------------------------------------------------------------

def predict_da360_disparity(
    panorama: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run DA360 and return **raw disparity** (not inverted depth).

    The DA360 model outputs scale-invariant disparity where higher values
    mean closer objects.  The SHARP alignment function expects this raw
    disparity — not the depth-converted output from ``DA360Model.predict()``.

    Args:
        panorama: (H, W, 3) uint8 ERP image.
        device: Torch device.

    Returns:
        (H, W) float32 disparity map (higher = closer).
    """
    import torch.nn.functional as F
    from .da360_model import DA360Model, DA360_INPUT_H, DA360_INPUT_W

    LOGGER.info("Loading DA360 model...")
    da360_model = DA360Model.load(device=device)

    # Prepare input — match DA360Model.predict() preprocessing
    image_tensor = torch.from_numpy(panorama.copy()).to(device)
    if image_tensor.dtype == torch.uint8:
        image_tensor = image_tensor.float() / 255.0
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    H, W = panorama.shape[:2]

    LOGGER.info("Running DA360 depth prediction...")
    with torch.inference_mode():
        x = F.interpolate(
            image_tensor,
            size=(DA360_INPUT_H, DA360_INPUT_W),
            mode="bilinear",
            align_corners=True,
        )
        output = da360_model.model(x)

    # Extract raw disparity
    if isinstance(output, dict):
        disparity = output.get("pred_disp", next(iter(output.values())))
    elif isinstance(output, (tuple, list)):
        disparity = output[0]
    else:
        disparity = output

    if disparity.dim() == 4:
        disparity = disparity.squeeze(1)
    if disparity.dim() == 3:
        disparity = disparity.squeeze(0)

    disparity = disparity.abs().clamp(min=1e-6)

    # Upsample to panorama resolution
    if disparity.shape[-2] != H or disparity.shape[-1] != W:
        disparity = F.interpolate(
            disparity.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        ).squeeze()

    result = disparity.float().cpu().numpy()
    LOGGER.info(
        "DA360 disparity: min=%.4f, max=%.4f, median=%.4f",
        float(result.min()), float(result.max()), float(np.median(result)),
    )
    return result


# ---------------------------------------------------------------------------
# SHARP predictor loading
# ---------------------------------------------------------------------------

def _load_sharp_predictor(device: torch.device):
    """Load the SHARP Gaussian predictor model.

    Auto-downloads the checkpoint from Apple CDN if not cached.

    Returns:
        (predictor, device) tuple.
    """
    from sharp.models import PredictorParams, create_predictor
    from sharp.cli.predict import DEFAULT_MODEL_URL

    LOGGER.info("Loading SHARP predictor (auto-download if needed)...")
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)

    return predictor


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def convert_sharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    side_count: int = 6,
    overlap_degrees: float = 10.0,
    seedvr2_upscale: bool = False,
    seedvr2_config: Optional["SeedVR2Config"] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Full SHARP 360 pipeline: ERP panorama -> merged 3DGS PLY.

    Args:
        input_path: Path to 2:1 equirectangular panorama image.
        output_path: Path for the output .ply file.
        device: Torch device ('cuda', 'mps', 'cpu').
        side_count: Number of horizon views (default 6).
        overlap_degrees: Extra FOV overlap per side in degrees.
        seedvr2_upscale: Whether to upscale faces with SeedVR2 first.
        seedvr2_config: SeedVR2Config (required if seedvr2_upscale=True).
        progress_callback: Optional callback(stage_name, current, total).

    Returns:
        Dict with stats: {"num_gaussians", "num_faces", "output_path"}.
    """
    from PIL import Image
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import Gaussians3D, apply_transform, save_ply

    def _progress(stage: str, cur: int, total: int):
        if progress_callback is not None:
            progress_callback(stage, cur, total)

    # ------------------------------------------------------------------
    # 1. Load and validate ERP panorama
    # ------------------------------------------------------------------
    _progress("load", 0, 1)
    LOGGER.info("Loading panorama: %s", input_path)
    pil_img = Image.open(input_path).convert("RGB")
    panorama = np.array(pil_img)
    erp_h, erp_w = panorama.shape[:2]

    if abs(erp_w / erp_h - 2.0) > 0.05:
        raise ValueError(
            f"Input image must be 2:1 equirectangular, got {erp_w}x{erp_h} "
            f"(ratio {erp_w / erp_h:.2f})"
        )
    LOGGER.info("Panorama: %dx%d", erp_w, erp_h)
    _progress("load", 1, 1)

    # ------------------------------------------------------------------
    # 2. Build extraction layout
    # ------------------------------------------------------------------
    face_size = min(erp_h, 1024)  # cap at 1024px for SHARP memory
    layout = build_extraction_layout(
        face_size=face_size,
        panorama_height=erp_h,
        side_count=side_count,
        overlap_degrees=overlap_degrees,
    )
    LOGGER.info(
        "Layout: %d views, %dx%d faces, focal=%.1f px",
        len(layout.views), layout.image_width, layout.image_height, layout.focal_px,
    )

    # ------------------------------------------------------------------
    # 3. Extract perspective faces
    # ------------------------------------------------------------------
    _progress("extract", 0, side_count)
    faces = extract_perspective_views(layout, panorama)
    _progress("extract", side_count, side_count)
    LOGGER.info("Extracted %d perspective faces", len(faces))

    # ------------------------------------------------------------------
    # 4. Optional SeedVR2 upscale
    # ------------------------------------------------------------------
    effective_width = layout.image_width
    effective_height = layout.image_height
    effective_focal_x = layout.focal_px
    effective_focal_y = layout.focal_y_px

    if seedvr2_upscale:
        from .seedvr2 import SeedVR2Config, upscale_images

        if seedvr2_config is None:
            seedvr2_config = SeedVR2Config()
        with tempfile.TemporaryDirectory(prefix="sharp360_seedvr2_") as tmp_dir:
            LOGGER.info("Upscaling %d faces with SeedVR2...", len(faces))
            _progress("seedvr2", 0, side_count)
            faces, new_w, new_h = upscale_images(faces, seedvr2_config, tmp_dir)
            _progress("seedvr2", side_count, side_count)

            # Update intrinsics for upscaled resolution
            scale_x = new_w / effective_width
            scale_y = new_h / effective_height
            effective_width = new_w
            effective_height = new_h
            effective_focal_x *= scale_x
            effective_focal_y *= scale_y
            LOGGER.info(
                "Upscaled to %dx%d, focal=(%.1f, %.1f)",
                effective_width, effective_height, effective_focal_x, effective_focal_y,
            )

    # ------------------------------------------------------------------
    # 5. DA360 disparity on full panorama
    # ------------------------------------------------------------------
    _progress("da360", 0, 1)
    da360_disparity = predict_da360_disparity(panorama, device)
    _progress("da360", 1, 1)
    LOGGER.info(
        "DA360 disparity: min=%.4f, max=%.4f, median=%.4f",
        da360_disparity.min(), da360_disparity.max(), np.median(da360_disparity),
    )

    # ------------------------------------------------------------------
    # 6. SHARP prediction per face
    # ------------------------------------------------------------------
    predictor = _load_sharp_predictor(device)
    f_px_tuple = (effective_focal_x, effective_focal_y)

    clip_degrees = 360.0 / side_count  # Total horizontal clip width per face

    all_gaussians: List[Gaussians3D] = []
    original_median_radii: List[float] = []

    for idx, view in enumerate(layout.views):
        _progress("sharp", idx, side_count)
        face_name = view.name
        face_img = faces[face_name]  # (H, W, 3) uint8

        LOGGER.info("SHARP predict: face %d/%d (%s)", idx + 1, side_count, face_name)

        # 6a. Run SHARP
        gaussians = predict_image(predictor, face_img, f_px_tuple, device)

        LOGGER.info(
            "  -> %d Gaussians before clipping",
            gaussians.mean_vectors.shape[1],
        )

        # ------------------------------------------------------------------
        # 7. Hard Voronoi border clipping (horizontal only)
        # ------------------------------------------------------------------
        gaussians = filter_gaussians_by_view_border(gaussians, clip_degrees)
        LOGGER.info(
            "  -> %d Gaussians after clipping (%.1f deg)",
            gaussians.mean_vectors.shape[1], clip_degrees,
        )

        # Store pre-alignment median radius for global restore later
        original_median_radii.append(float(torch.median(
            torch.norm(gaussians.mean_vectors, dim=-1),
        ).item()))

        # ------------------------------------------------------------------
        # 8. DA360 alignment
        # ------------------------------------------------------------------
        ref_disp_view = extract_perspective_scalar_view(
            da360_disparity,
            effective_width,
            effective_height,
            effective_focal_x,
            effective_focal_y,
            view,
        )
        gaussians, med_scale, n_samples = align_gaussians_to_reference(
            gaussians,
            ref_disp_view,
            effective_focal_x,
            effective_focal_y,
            effective_width,
            effective_height,
            grid_resolution=8,
        )
        LOGGER.info(
            "  Aligned %s: median_scale=%.4f (%d samples)",
            view.name, med_scale, n_samples,
        )

        # ------------------------------------------------------------------
        # 9. Rotate into world frame
        # ------------------------------------------------------------------
        R = view.rotation_matrix  # 3x3
        transform = torch.zeros(3, 4, device=device, dtype=torch.float32)
        transform[:3, :3] = torch.from_numpy(R).float().to(device)
        # Translation is zero (viewer at origin)

        gaussians = apply_transform(gaussians, transform)

        all_gaussians.append(gaussians)
        LOGGER.info(
            "  -> %d Gaussians in world frame",
            gaussians.mean_vectors.shape[1],
        )

    _progress("sharp", side_count, side_count)

    # ------------------------------------------------------------------
    # 10. Merge all faces
    # ------------------------------------------------------------------
    merged = merge_gaussians(all_gaussians)
    total_gaussians = merged.mean_vectors.shape[1]
    LOGGER.info("Merged: %d total Gaussians from %d faces", total_gaussians, side_count)

    # ------------------------------------------------------------------
    # 11. Global scale restore
    # ------------------------------------------------------------------
    # Restore median scene radius to the pre-alignment value so alignment
    # doesn't shrink/grow the overall scene (matches SHARP_360_to_Splat reference)
    if original_median_radii:
        original_scene_median = float(np.median(original_median_radii))
        current_median = float(torch.median(
            torch.norm(merged.mean_vectors, dim=-1),
        ).item())
        if current_median > 1e-8:
            global_scale = original_scene_median / current_median
            merged = scale_gaussians(merged, global_scale)
            LOGGER.info(
                "Global scale restore: %.4f (%.2f -> %.2f median radius)",
                global_scale, current_median, original_scene_median,
            )

    # ------------------------------------------------------------------
    # 11.5. Flip Y to match viewer convention
    # ------------------------------------------------------------------
    # SHARP uses Y-down camera convention (down = +Y in world frame).
    # The GaussianSplats3D viewer and SPAG's DA360 path use Y-up.
    # Negate Y to convert from SHARP's Y-down world to the viewer's Y-up.
    from sharp.utils.gaussians import Gaussians3D as _G3D
    flipped_means = merged.mean_vectors.clone()
    flipped_means[:, :, 1] *= -1.0
    merged = _G3D(
        mean_vectors=flipped_means,
        singular_values=merged.singular_values,
        quaternions=merged.quaternions,
        colors=merged.colors,
        opacities=merged.opacities,
    )

    # ------------------------------------------------------------------
    # 12. PLY export via SHARP's save_ply()
    # ------------------------------------------------------------------
    _progress("export", 0, 1)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_ply(
        merged,
        f_px=f_px_tuple,
        image_shape=(effective_height, effective_width),
        path=out_path,
    )
    LOGGER.info("Saved PLY: %s (%d Gaussians)", out_path, total_gaussians)
    _progress("export", 1, 1)

    return {
        "num_gaussians": total_gaussians,
        "num_faces": side_count,
        "output_path": str(out_path),
    }
