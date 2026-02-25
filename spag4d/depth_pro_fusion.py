# spag4d/depth_pro_fusion.py
"""
Phase 1: Cubemap Depth Pro Fusion for SPAG-4D.

Projects a 360° equirectangular panorama onto 6 cubemap faces, runs
Apple Depth Pro on each face for metric depth, aligns the per-face depths
to the globally-consistent DAP/PanDA depth, then composites back to ERP
via feathered (edge-distance) blending.

The key improvement over DAP-only depth: Depth Pro produces sharper object
boundaries and better small-scale geometry, while DAP provides global
scale/shift consistency.  The Laplacian pyramid fusion (Phase 4) combines
both strengths after this step.

Requires apple/ml-depth-pro (loaded lazily; absent means graceful no-op):
    git clone https://github.com/apple/ml-depth-pro src/ml-depth-pro
    pip install -e src/ml-depth-pro
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class ProjectionMode(Enum):
    CUBEMAP     = "cubemap"      # 6 axis-aligned faces — fast, default
    ICOSAHEDRON = "icosahedron"  # 20 faces — higher quality (reserved for future)


# ──────────────────────────────────────────────────────────────────────────────
# Cubemap face axis definitions
#
# SPAG-4D ERP convention (from spherical_grid.py):
#   x = sin(phi) * cos(theta)
#   y = cos(phi)                 (y = UP, phi=0 at north pole)
#   z = -sin(phi) * sin(theta)
#
# Face configs: (forward, right, up) unit vectors in this coordinate system.
# "right" = cross(forward, world_up) for equatorial faces;
#  up/down faces use a consistent tangent-space convention.
# ──────────────────────────────────────────────────────────────────────────────

_FACE_AXES: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    # name    forward             camera-right        camera-up
    "front":  (np.array([ 1.,  0.,  0.]),  np.array([ 0.,  0., -1.]),  np.array([ 0.,  1.,  0.])),
    "back":   (np.array([-1.,  0.,  0.]),  np.array([ 0.,  0.,  1.]),  np.array([ 0.,  1.,  0.])),
    "right":  (np.array([ 0.,  0.,  1.]),  np.array([-1.,  0.,  0.]),  np.array([ 0.,  1.,  0.])),
    "left":   (np.array([ 0.,  0., -1.]),  np.array([ 1.,  0.,  0.]),  np.array([ 0.,  1.,  0.])),
    "up":     (np.array([ 0.,  1.,  0.]),  np.array([ 1.,  0.,  0.]),  np.array([ 0.,  0.,  1.])),
    "down":   (np.array([ 0., -1.,  0.]),  np.array([ 1.,  0.,  0.]),  np.array([ 0.,  0., -1.])),
}


# ──────────────────────────────────────────────────────────────────────────────
# Pure-math projection helpers (no model dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _xyz_to_erp_coords(
    xyz: np.ndarray,   # (..., 3) — may be unnormalised
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 3-D sphere points to equirectangular pixel coordinates.

    Returns:
        rows: (...) float in [0, H)
        cols: (...) float in [0, W)
    """
    eps = 1e-9
    r    = np.sqrt((xyz ** 2).sum(axis=-1)).clip(eps)
    y_n  = (xyz[..., 1] / r).clip(-1.0, 1.0)
    phi  = np.arccos(y_n)                                        # [0, π]

    theta_raw = np.arctan2(-xyz[..., 2], xyz[..., 0])            # (−π, π]
    theta = np.where(theta_raw < 0, theta_raw + 2.0 * np.pi, theta_raw)  # [0, 2π)

    rows = phi   / np.pi          * H
    cols = (1.0 - theta / (2.0 * np.pi)) * W
    cols = cols % W                                              # wrap seam
    return rows.astype(np.float32), cols.astype(np.float32)


def _build_face_rays(
    face_name: str,
    face_size: int,
    fov_deg: float,
) -> np.ndarray:
    """
    Compute unit-sphere ray directions for every pixel of one cubemap face.

    Returns:
        rays: (face_size, face_size, 3) float32 unit vectors
    """
    if face_size < 2:
        raise ValueError(f"face_size must be >= 2, got {face_size}")

    forward, right, up = _FACE_AXES[face_name]
    t = np.tan(np.radians(fov_deg / 2.0))

    j = np.arange(face_size, dtype=np.float64)   # col index
    i = np.arange(face_size, dtype=np.float64)   # row index
    jj, ii = np.meshgrid(j, i)

    # u: column → camera-right axis; v: row=0 is top → camera-up axis
    u = (2.0 * jj / (face_size - 1) - 1.0) * t   # [−t, +t]
    v = (1.0 - 2.0 * ii / (face_size - 1)) * t   # [+t, −t]  (row 0 = top = +up)

    rays = (
        forward[None, None, :]
        + u[..., None] * right[None, None, :]
        + v[..., None] * up[None, None, :]
    )
    norms = np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-9)
    return (rays / norms).astype(np.float32)


def project_erp_to_face(
    erp_image: np.ndarray,   # (H, W) or (H, W, C)
    face_name: str,
    face_size: int = 512,
    fov_deg: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample an equirectangular image onto a single cubemap face.

    Args:
        erp_image:  (H, W[, C]) float32 or uint8
        face_name:  "front" | "back" | "right" | "left" | "up" | "down"
        face_size:  output resolution (pixels per side)
        fov_deg:    field of view in degrees (100° gives good seam overlap)

    Returns:
        face_image: (face_size, face_size[, C]) — same dtype as input
        uv_map:     (face_size, face_size, 2) float32 ERP coords (col, row)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required.  pip install opencv-python")

    if face_name not in _FACE_AXES:
        raise ValueError(f"Unknown face: {face_name!r}.  Choose from {list(_FACE_AXES)}")

    H, W = erp_image.shape[:2]
    rays = _build_face_rays(face_name, face_size, fov_deg)   # (F, F, 3)
    rows, cols = _xyz_to_erp_coords(rays, H, W)              # (F, F)

    # cv2.remap convention: map_x = col, map_y = row
    face_image = cv2.remap(
        erp_image.astype(np.float32) if erp_image.dtype not in (np.uint8, np.float32)
        else erp_image,
        cols,          # map_x
        rows,          # map_y
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    if erp_image.dtype == np.uint8:
        face_image = face_image.clip(0, 255).astype(np.uint8)

    uv_map = np.stack([cols, rows], axis=-1)   # (F, F, 2) — (col, row)
    return face_image, uv_map


def project_erp_to_cubemap(
    erp_image: np.ndarray,
    face_size: int = 512,
    fov_deg: float = 100.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Project ERP to all 6 cubemap faces.

    Returns:
        face_images: {face_name: (F, F[, C])}
        uv_maps:     {face_name: (F, F, 2)}  ERP coords (col, row)
    """
    face_images: Dict[str, np.ndarray] = {}
    uv_maps: Dict[str, np.ndarray] = {}
    for face_name in _FACE_AXES:
        img, uv = project_erp_to_face(erp_image, face_name, face_size, fov_deg)
        face_images[face_name] = img
        uv_maps[face_name] = uv
    return face_images, uv_maps


def align_depth_to_reference(
    face_depth: np.ndarray,    # (F, F) float32 — metric depth from Depth Pro
    face_uv_map: np.ndarray,   # (F, F, 2) — ERP coords (col, row)
    dap_depth: np.ndarray,     # (H, W) float32 — globally-consistent reference
    min_valid_pixels: int = 50,
) -> np.ndarray:
    """
    Scale-align `face_depth` to the DAP reference in log-depth space.

    Finds scalar s such that s * face_depth ≈ dap_depth at valid pixels.
    Uses: s = exp(median(log(dap) − log(face_depth)))

    Returns:
        aligned: (F, F) float32 depth in DAP metric units
    """
    H, W = dap_depth.shape
    erp_col = np.round(face_uv_map[..., 0]).astype(int).clip(0, W - 1)
    erp_row = np.round(face_uv_map[..., 1]).astype(int).clip(0, H - 1)

    dap_at_face = dap_depth[erp_row, erp_col].astype(np.float64)
    dp_flat     = face_depth.astype(np.float64)

    valid = (dap_at_face > 1e-3) & (dp_flat > 1e-3)
    if valid.sum() < min_valid_pixels:
        warnings.warn(
            f"Only {valid.sum()} valid pixels for depth alignment "
            "(need ≥ {min_valid_pixels}); returning unscaled Depth Pro depth."
        )
        return face_depth.clip(0.0, None).astype(np.float32)

    log_scale = np.median(np.log(dap_at_face[valid]) - np.log(dp_flat[valid]))
    scale = float(np.exp(np.clip(log_scale, -3.0, 3.0)))   # max ×20 correction
    return (face_depth * scale).clip(0.0, None).astype(np.float32)


def composite_faces_to_erp(
    face_depths: Dict[str, np.ndarray],      # {face_name: (F, F) float32}
    uv_maps: Dict[str, np.ndarray],          # {face_name: (F, F, 2)}
    erp_shape: Tuple[int, int],              # (H, W)
    dap_depth: Optional[np.ndarray] = None, # (H, W) fallback for gaps
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite per-face depth maps back to ERP using feathered blending.

    Each face pixel is weighted by its distance from the face edge
    (centre = max weight, edge = 0), preventing seam artefacts.

    Returns:
        depth:      (H, W) float32 blended depth
        confidence: (H, W) float32 in [0, 1] (normalised accumulated weight)
    """
    H, W = erp_shape
    depth_acc  = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)

    for face_name, depth in face_depths.items():
        uv = uv_maps[face_name]
        F  = depth.shape[0]

        # Linear edge-distance weight: 1 at centre, 0 at border pixels
        idx = np.arange(F, dtype=np.float64)
        u   = np.minimum(idx, idx[::-1])
        uu, vv = np.meshgrid(u / max(F / 2.0, 1.0), u / max(F / 2.0, 1.0))
        w = np.minimum(uu, vv).clip(0.0, 1.0)

        erp_col = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
        erp_row = np.round(uv[..., 1]).astype(int).clip(0, H - 1)

        np.add.at(depth_acc,  (erp_row, erp_col), depth.astype(np.float64) * w)
        np.add.at(weight_acc, (erp_row, erp_col), w)

    safe_w     = np.where(weight_acc > 1e-8, weight_acc, 1.0)
    composite  = (depth_acc / safe_w).clip(0.0, None).astype(np.float32)
    confidence = (weight_acc / max(float(weight_acc.max()), 1e-8)).astype(np.float32)

    if dap_depth is not None:
        gaps = weight_acc < 1e-8
        composite[gaps]  = dap_depth.astype(np.float32)[gaps]
        confidence[gaps] = 0.0

    return composite, confidence


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class DepthProFusion:
    """
    Fuse Apple Depth Pro face predictions with DAP panorama depth.

    Usage::

        fusion = DepthProFusion(device, face_size=512)
        fusion.load_model()
        fused_depth, confidence = fusion.fuse(erp_image_np, dap_depth_np)

    The returned `confidence` map is suitable as the `confidence` argument to
    `DepthBlender.fuse()` in Phase 4, letting the Laplacian pyramid prefer
    Depth Pro where it is reliable and DAP elsewhere.
    """

    def __init__(
        self,
        device,
        face_size: int = 512,
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
        self._transform      = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Lazy-load Apple Depth Pro.

        Requires ml-depth-pro to be installed::

            git clone https://github.com/apple/ml-depth-pro src/ml-depth-pro
            pip install -e src/ml-depth-pro
        """
        if self._model is not None:
            return

        path = model_path or self.model_path
        try:
            import depth_pro  # type: ignore
            kwargs: dict = {}
            if path:
                kwargs["pretrained_resource"] = f"local://{path}"
            self._model, self._transform = depth_pro.create_model_and_transforms(**kwargs)
            self._model = self._model.to(self.device).eval()
        except ImportError:
            raise ImportError(
                "Apple Depth Pro is not installed.  Run:\n"
                "  git clone https://github.com/apple/ml-depth-pro src/ml-depth-pro\n"
                "  pip install -e src/ml-depth-pro"
            )

    def fuse(
        self,
        erp_image: np.ndarray,   # (H, W, 3) uint8
        dap_depth: np.ndarray,   # (H, W) float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full fusion pipeline: project → infer → align → composite.

        Args:
            erp_image:  (H, W, 3) uint8 equirectangular RGB panorama
            dap_depth:  (H, W) float32 globally-consistent depth (DAP/PanDA)

        Returns:
            fused_depth: (H, W) float32 in DAP metric space
            confidence:  (H, W) float32 in [0, 1] (1 = fully Depth Pro)
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded.  Call DepthProFusion.load_model() first."
            )

        if self.projection_mode == ProjectionMode.ICOSAHEDRON:
            warnings.warn(
                "ICOSAHEDRON projection is not yet implemented; "
                "falling back to CUBEMAP."
            )

        H, W = erp_image.shape[:2]

        # 1. Project ERP → 6 cubemap faces
        face_images, uv_maps = project_erp_to_cubemap(
            erp_image, face_size=self.face_size, fov_deg=self.fov_deg
        )

        # 2. Run Depth Pro on each face
        raw_depths = self._run_depth_pro(face_images)

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

    def _run_depth_pro(
        self,
        face_images: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Run Apple Depth Pro on each face image.

        Returns:
            {face_name: (F, F) float32 depth in metres}
        """
        import torch
        from PIL import Image as PILImage

        face_depths: Dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for face_name, face_img in face_images.items():
                # Ensure uint8 for PIL
                if face_img.dtype != np.uint8:
                    face_img = face_img.clip(0, 255).astype(np.uint8)
                pil_img = PILImage.fromarray(face_img)

                transformed = self._transform(pil_img)
                prediction  = self._model.infer(transformed)

                depth_t = prediction["depth"]   # (H, W) or (1, H, W)
                if depth_t.ndim == 3:
                    depth_t = depth_t[0]
                face_depths[face_name] = depth_t.cpu().numpy().astype(np.float32)

        return face_depths
