# spag4d/sharp_gaussian_pipeline.py
"""
Direct SHARP Gaussian pipeline for 360° splat generation.

Instead of using a geometric ERP-grid converter with SHARP attribute refinement,
this module runs SHARP on cubemap/icosahedral faces and uses its native Gaussian
output directly.  Each face's Gaussians are depth-aligned to a DAP reference map
and rotated into a common world frame.

Pipeline:
  ERP image → DAP depth (metric reference)
            → project to cubemap faces
            → SHARP on each face → native Gaussians3D per face
            → per-face DAP depth alignment
            → rotate face-local Gaussians to world Y-up frame
            → nearest-face ownership (no double-coverage)
            → merge faces
            → global scale consistency correction
            → seam color smoothing
            → output 360° splat dict
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from pathlib import Path



def _build_face_rotation(right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> torch.Tensor:
    """Build 3×3 rotation matrix from face-camera axes to world frame.

    SHARP outputs positions as (cam_right, cam_up, cam_forward).
    The rotation R transforms those into world Y-up coordinates:
        p_world = R @ p_camera
    """
    R = np.column_stack([right, up, forward]).astype(np.float32)
    return torch.from_numpy(R)


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3×3 rotation matrix to quaternion (WXYZ order).

    Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / q.norm()


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication.  q1, q2 in WXYZ order.  Batched over q2's dim 0."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2 = q2[:, 0]; x2 = q2[:, 1]; y2 = q2[:, 2]; z2 = q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _nearest_face_mask(
    positions_world: torch.Tensor,
    face_idx: int,
    all_face_forwards: List[np.ndarray],
) -> torch.Tensor:
    """Boolean mask: True for Gaussians whose nearest face center is *this* face.

    Each Gaussian is assigned to a single face — no double-coverage.
    """
    dirs = F.normalize(positions_world, dim=-1)
    angles = []
    for fwd in all_face_forwards:
        fwd_t = torch.tensor(fwd, dtype=dirs.dtype, device=dirs.device)
        cos_a = (dirs * fwd_t).sum(dim=-1)
        angles.append(torch.acos(cos_a.clamp(-1, 1)))
    angles = torch.stack(angles, dim=-1)  # [N, num_faces]
    nearest = angles.argmin(dim=-1)        # [N]
    return nearest == face_idx


def _compute_face_weight(
    positions_world: torch.Tensor,
    face_idx: int,
    all_face_forwards: List[np.ndarray],
    face_fov: float,
    overlap_ratio: float,
) -> torch.Tensor:
    """Per-Gaussian soft weight for face ownership.

    Mirrors projection.py's 3-component blending strategy:
      weight = gaussian_decay × cosine²_taper × in_fov

    Core zone (angle < base_half_angle): weight ≈ 1.0
    Taper zone (base_half_angle → half_fov): smooth cosine² fade
    Beyond half_fov: weight = 0.0

    Works for both cubemap (6 faces) and icosahedral (20 faces).

    Returns:
        Float tensor [N] in [0, 1].
    """
    dirs = F.normalize(positions_world, dim=-1)
    fwd = torch.tensor(
        all_face_forwards[face_idx], dtype=dirs.dtype, device=dirs.device,
    )

    cos_angle = (dirs * fwd).sum(dim=-1).clamp(-1, 1)
    angle = torch.acos(cos_angle)

    half_fov = face_fov / 2.0
    # base_half_angle: the angular extent of the "core" zone (no taper)
    # Matches projection.py: base_half = 1.0 / (1.0 + overlap_ratio)
    base_half_angle = half_fov / (1.0 + overlap_ratio)
    taper_range = max(half_fov - base_half_angle, 1e-6)

    # Cosine² taper: 1.0 at base_half_angle, 0.0 at half_fov
    taper_dist = ((angle - base_half_angle) / taper_range).clamp(0, 1)
    taper = torch.cos(taper_dist * (math.pi / 2)) ** 2

    # Gaussian angular decay (smooth falloff from face center)
    gaussian_w = torch.exp(-angle ** 2 / (face_fov ** 2 / 4))

    in_fov = angle < half_fov
    weight = gaussian_w * taper * in_fov.float()
    return weight


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class SHARPGaussianPipeline:
    """
    Generate a 360° Gaussian Splat directly from SHARP's native Gaussian output.

    This bypasses the geometric ERP-grid converter entirely and uses SHARP's
    trained Gaussian prediction from each cubemap face.
    """

    SHARP_WEIGHTS_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    CACHE_DIR = Path.home() / ".cache" / "spag4d" / "sharp"
    SHARP_INPUT_SIZE = 1536

    def __init__(
        self,
        device: torch.device,
        cubemap_size: int = 1536,
        projection_mode: str = "cubemap",
    ):
        self.device = device
        self.cubemap_size = cubemap_size
        self.projection_mode = projection_mode
        self.model = None
        self.projector = None

    # -- Model loading --

    def load_model(self, model_path: Optional[str] = None):
        if self.model is not None:
            return
        print("[SHARP-Direct] Loading SHARP model...")
        t0 = time.time()

        try:
            from sharp.models import create_predictor, PredictorParams
        except ImportError as e:
            raise ImportError(
                "ML-SHARP module not found. Install with:\n"
                "pip install --no-deps https://github.com/apple/ml-sharp/archive/refs/heads/main.zip"
            ) from e

        if model_path is None:
            model_path = self._get_or_download_weights()

        params = PredictorParams()
        self._configure_max_quality(params)
        self.model = create_predictor(params)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            state_dict = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        self.projector = None  # will init lazily
        print(f"[SHARP-Direct] Model loaded in {time.time() - t0:.1f}s")

    @staticmethod
    def _configure_max_quality(params):
        """Configure PredictorParams for maximum output quality."""
        try:
            params.low_pass_filter_eps = 0.001
            params.max_scale = 10.0
            params.min_scale = 0.0

            # linearRGB — Apple's default, what SHARP was trained with.
            # PLY writer applies linearRGB->sRGB conversion on export.
            params.color_space = "linearRGB"

            params.initializer.scale_factor = 1.0
            params.initializer.color_option = "all_layers"
            params.initializer.first_layer_depth_option = "surface_min"
            params.initializer.rest_layer_depth_option = "surface_min"
            params.initializer.normalize_depth = True

            params.delta_factor.xy = 0.001
            params.delta_factor.z = 0.001
            params.delta_factor.color = 0.1  # Apple's default for linearRGB
            params.delta_factor.opacity = 1.0
            params.delta_factor.scale = 1.0
            params.delta_factor.quaternion = 1.0

            params.monodepth.use_patch_overlap = True
            params.gaussian_decoder.use_depth_input = True
        except AttributeError as e:
            import warnings
            warnings.warn(f"Could not set some SHARP quality parameters: {e}")

    def _get_or_download_weights(self) -> str:
        """Download SHARP weights if not cached."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self.CACHE_DIR / "sharp.pt"

        if cache_path.exists():
            return str(cache_path)

        print("Downloading SHARP weights...")
        try:
            import urllib.request
            import ssl

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(self.SHARP_WEIGHTS_URL, context=ctx) as response, \
                 open(cache_path, 'wb') as out_file:
                downloaded = 0
                block_size = 8192
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if downloaded % (1024 * 1024) == 0:
                        print(f"Downloading: {downloaded / 1024 / 1024:.1f} MB", end='\r')

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

    def _init_projector(self):
        if self.projector is not None:
            return
        from .projection import get_projector
        self.projector = get_projector(
            mode=self.projection_mode,
            face_size=self.cubemap_size,
            device=self.device,
        )

    # -- Core pipeline --

    @torch.inference_mode()
    def generate(
        self,
        erp_image: torch.Tensor,       # [H, W, 3] float [0,1] or uint8
        dap_depth: torch.Tensor,        # [H, W] metric depth from DAP
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        sky_threshold: float = 80.0,
        grid_jitter: float = 0.03,
    ) -> dict:
        """
        Generate 360° Gaussians from SHARP's native output.

        Args:
            erp_image: equirectangular RGB image
            dap_depth: metric depth map from DAP (global reference)
            depth_min: minimum valid depth
            depth_max: maximum valid depth
            sky_threshold: Gaussians beyond this distance are discarded
            grid_jitter: sub-pixel jitter intensity (0=off, 0.5=max).
                Breaks SHARP's regular 768x768 grid Moiré. Default 0.03
                (subtle anti-aliasing, barely visible).

        Returns:
            dict with means, scales, quats, colors, opacities (flat tensors)
        """
        t_total = time.time()
        if self.model is None:
            self.load_model()
        self._init_projector()

        H, W = erp_image.shape[:2]
        print(f"[SHARP-Direct] Starting generate: {W}x{H} image, "
              f"mode={self.projection_mode}, cubemap_size={self.cubemap_size}")

        # Convert to uint8 numpy for projection
        if erp_image.dtype == torch.uint8:
            erp_np = erp_image.cpu().numpy()
        else:
            erp_np = (erp_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        dap_np = dap_depth.cpu().numpy().astype(np.float32)

        # Project ERP → faces
        t0 = time.time()
        faces_list = self.projector.project_erp_to_faces(erp_np)
        num_faces = len(faces_list)
        print(f"[SHARP-Direct] Projected ERP to {num_faces} faces in {time.time() - t0:.1f}s")

        # Focal length for disparity_factor
        f_px = self.cubemap_size / (2 * math.tan(self.projector.face_fov / 2))

        # Build face rotation matrices and collect forward vectors for overlap
        face_rotations = []  # List of (R_3x3, q_wxyz, forward)
        all_face_forwards = []  # forward vectors for feathered blending
        for i in range(num_faces):
            fwd = np.asarray(self.projector.face_directions[i], dtype=np.float32)
            up_raw = np.asarray(self.projector.face_ups[i], dtype=np.float32)
            right = np.cross(up_raw, fwd)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, fwd)
            up = up / (np.linalg.norm(up) + 1e-8)
            R = _build_face_rotation(right, up, fwd)
            q = _rotation_matrix_to_quaternion(R)
            face_rotations.append((R.to(self.device), q.to(self.device), fwd))
            all_face_forwards.append(fwd)

        # Process each face
        all_means = []
        all_scales = []
        all_quats = []
        all_colors = []
        all_opacities = []
        all_face_labels = []
        face_scale_factors = []

        for face_idx in range(num_faces):
            t_face = time.time()
            face_np = faces_list[face_idx]
            face_t = torch.from_numpy(face_np).float().to(self.device) / 255.0

            # --- Run SHARP ---
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: running SHARP inference...", flush=True)
            t0 = time.time()
            gaussians = self._run_sharp(face_t, f_px)
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: SHARP inference done in {time.time() - t0:.1f}s")

            # --- Extract Gaussians (both layers) ---
            grid_size = self.SHARP_INPUT_SIZE // 2  # 768
            means_cam, scales_cam, quats_cam, colors, opacities = (
                self._unpack_gaussians(gaussians, grid_size)
            )
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: unpacked {means_cam.shape[0]:,} Gaussians (front+back)")
            # means_cam: [N, 3] in face-camera metric coords (right, up, forward)
            # quats_cam: [N, 4] WXYZ in face-camera frame

            del gaussians
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # --- Align depth to DAP ---
            t0 = time.time()
            means_cam, face_sf = self._align_to_dap(
                means_cam, face_idx, dap_np, H, W, scales_cam
            )
            face_scale_factors.append(face_sf)
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: DAP alignment done in {time.time() - t0:.1f}s")

            # --- Grid jitter (break regular dot pattern) ---
            if grid_jitter > 0:
                depth_j = means_cam[:, 2].clamp(min=0.1)
                pixel_spacing = 2.0  # SHARP outputs at half input resolution
                spacing_cam = depth_j * pixel_spacing / f_px
                rng = torch.Generator(device=means_cam.device)
                rng.manual_seed(42 + face_idx)
                jitter_offset = torch.randn(
                    means_cam.shape[0], 2,
                    device=means_cam.device, generator=rng,
                ) * (spacing_cam * grid_jitter).unsqueeze(-1)
                means_cam = means_cam.clone()
                means_cam[:, 0] += jitter_offset[:, 0]  # right
                means_cam[:, 1] += jitter_offset[:, 1]  # up

            # --- Filter invalid ---
            cam_depth = means_cam[:, 2]  # forward component = depth
            valid = (cam_depth > depth_min) & (cam_depth < depth_max)
            means_cam = means_cam[valid]
            scales_cam = scales_cam[valid]
            quats_cam = quats_cam[valid]
            colors = colors[valid]
            opacities = opacities[valid]

            n_valid = means_cam.shape[0]
            if n_valid == 0:
                print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: 0 valid Gaussians, skipping")
                continue

            # --- Transform to world frame ---
            R_world, q_world, fwd = face_rotations[face_idx]
            # Position: p_world = R @ p_cam
            means_world = (R_world @ means_cam.T).T  # [N, 3]
            # Quaternion: compose face rotation with per-Gaussian rotation
            quats_world = _quat_multiply(q_world, quats_cam)
            # Scales are invariant to rotation (eigen-values don't change)

            # --- Nearest-face ownership (prevents double-coverage ghosting) ---
            # Each Gaussian is assigned to exactly one face — the one whose
            # center is closest to the Gaussian's direction.  Feathered
            # blending was tried but creates ghosting because two faces
            # produce DIFFERENT Gaussians at different positions; making
            # both semi-transparent doesn't blend them, it doubles them.
            own = _nearest_face_mask(
                means_world, face_idx, all_face_forwards,
            )

            # --- Sky filter ---
            dist = means_world.norm(dim=-1)
            if sky_threshold > 0:
                keep = own & (dist < sky_threshold)
            else:
                keep = own  # sky_threshold=0 disables sky filtering
            means_world = means_world[keep]
            scales_cam = scales_cam[keep]
            quats_world = quats_world[keep]
            colors = colors[keep]
            opacities = opacities[keep]

            n_kept = means_world.shape[0]
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: {n_valid:,} valid -> {n_kept:,} after ownership+sky ({time.time() - t_face:.1f}s)")

            all_means.append(means_world)
            all_scales.append(scales_cam)
            all_quats.append(quats_world)
            all_colors.append(colors)
            all_opacities.append(opacities)
            all_face_labels.append(
                torch.full((means_world.shape[0],), face_idx,
                           dtype=torch.long, device=means_world.device)
            )

        # --- Merge faces ---
        if not all_means:
            print("[SHARP-Direct] WARNING: No faces produced valid Gaussians.")
            return {
                'means': torch.zeros((0, 3), device=self.device),
                'scales': torch.zeros((0, 3), device=self.device),
                'quats': torch.zeros((0, 4), device=self.device),
                'colors': torch.zeros((0, 3), device=self.device),
                'opacities': torch.zeros((0, 1), device=self.device),
            }

        print(f"[SHARP-Direct] Merging {num_faces} faces...")
        means = torch.cat(all_means, dim=0)
        scales = torch.cat(all_scales, dim=0)
        quats = torch.cat(all_quats, dim=0)
        colors = torch.cat(all_colors, dim=0)
        opacities = torch.cat(all_opacities, dim=0).clamp(0.01, 0.99).unsqueeze(-1)
        total_gaussians = means.shape[0]
        print(f"[SHARP-Direct] Merged: {total_gaussians:,} total Gaussians")

        # --- Overlap-based scale alignment ---
        # Per-face DAP alignment gives each face a scale factor, but these
        # can diverge wildly (e.g. 3.8x for ground vs 20x for sky).
        # Instead of trusting DAP, use SHARP's own overlap zones: where two
        # faces see the same scene geometry, compare their radial distances
        # to compute relative scale corrections. This is SHARP-to-SHARP
        # comparison — far more reliable than SHARP-to-DAP.
        face_labels = torch.cat(all_face_labels, dim=0)
        if len(face_scale_factors) > 1:
            print(f"[SHARP-Direct] DAP scale factors: "
                  f"[{', '.join(f'{s:.2f}' for s in face_scale_factors)}]")

            t0 = time.time()
            corrections = self._overlap_scale_corrections(
                means, face_labels, all_face_forwards, num_faces,
            )
            print(f"[SHARP-Direct] Overlap alignment: corrections="
                  f"[{', '.join(f'{c:.3f}' for c in corrections)}] "
                  f"({time.time() - t0:.1f}s)")

            for fi in range(num_faces):
                if abs(corrections[fi] - 1.0) > 0.005:
                    mask = face_labels == fi
                    n_affected = mask.sum().item()
                    if n_affected > 0:
                        means[mask] *= corrections[fi]
                        scales[mask] *= corrections[fi]
                        print(f"[SHARP-Direct]   Face {fi}: x{corrections[fi]:.3f} "
                              f"({n_affected:,} Gaussians)")

        # --- Post-merge seam color smoothing ---
        if len(all_face_labels) > 1:
            print(f"[SHARP-Direct] Running seam color smoothing...", flush=True)
            t0 = time.time()
            colors = self._smooth_seam_colors(
                means, colors, face_labels, all_face_forwards,
            )
            print(f"[SHARP-Direct] Seam smoothing done in {time.time() - t0:.1f}s")

        # Convert quaternion from WXYZ (SHARP) to XYZW (3DGS convention)
        quats_xyzw = torch.cat([quats[:, 1:4], quats[:, 0:1]], dim=-1)

        print(f"[SHARP-Direct] Generate complete: {total_gaussians:,} Gaussians in {time.time() - t_total:.1f}s")
        return {
            'means': means,
            'scales': scales,
            'quats': quats_xyzw,
            'colors': colors,
            'opacities': opacities,
        }

    # -- Helpers --

    def _run_sharp(self, face: torch.Tensor, f_px: float):
        """Run SHARP on one face image.  Returns Gaussians3D."""
        face_batch = face.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        if face_batch.shape[-1] != self.SHARP_INPUT_SIZE:
            face_batch = F.interpolate(
                face_batch,
                size=(self.SHARP_INPUT_SIZE, self.SHARP_INPUT_SIZE),
                mode='bilinear', align_corners=False,
            )
        disparity_factor = torch.tensor(
            [f_px / self.cubemap_size],
            device=self.device, dtype=torch.float32,
        )
        return self.model(face_batch, disparity_factor=disparity_factor)

    def _unpack_gaussians(self, gaussians, grid_size: int):
        """Unpack Gaussians3D into flat tensors, blending dual layers."""
        N_layer = grid_size * grid_size
        # mean_vectors: [1, N_layer*2, 3]  (batch=1, 2 layers flattened)
        means_all = gaussians.mean_vectors[0]   # [N_layer*2, 3]
        scales_all = gaussians.singular_values[0]
        quats_all = gaussians.quaternions[0]     # WXYZ
        colors_all = gaussians.colors[0]
        opas_all = gaussians.opacities[0]        # [N_layer*2]

        # Reshape to [2, grid, grid, ...]
        means  = means_all.view(2, grid_size, grid_size, 3)
        scales = scales_all.view(2, grid_size, grid_size, 3)
        quats  = quats_all.view(2, grid_size, grid_size, 4)
        colors = colors_all.view(2, grid_size, grid_size, 3)
        opas   = opas_all.view(2, grid_size, grid_size)

        # Use front layer (0) as primary; include back layer (1) where
        # opacity indicates useful geometry (thin structures, foliage).
        front_opa = opas[0]
        back_opa = opas[1]
        # Keep back-layer Gaussians only where they have high-confidence opacity.
        # Low threshold (0.1) keeps ~90% of back-layer which adds floaters
        # and duplicates in overlap zones. 0.3 keeps only confident geometry.
        back_useful = back_opa > 0.3

        # Front layer: always keep
        m_front = means[0].reshape(-1, 3)
        s_front = scales[0].reshape(-1, 3)
        q_front = quats[0].reshape(-1, 4)
        c_front = colors[0].reshape(-1, 3)
        o_front = front_opa.reshape(-1)

        # Back layer: keep only useful ones
        mask_back = back_useful.reshape(-1)
        m_back = means[1].reshape(-1, 3)[mask_back]
        s_back = scales[1].reshape(-1, 3)[mask_back]
        q_back = quats[1].reshape(-1, 4)[mask_back]
        c_back = colors[1].reshape(-1, 3)[mask_back]
        o_back = back_opa.reshape(-1)[mask_back]

        return (
            torch.cat([m_front, m_back], dim=0),
            torch.cat([s_front, s_back], dim=0),
            torch.cat([q_front, q_back], dim=0),
            torch.cat([c_front, c_back], dim=0).clamp(0, 1),
            torch.cat([o_front, o_back], dim=0),
        )

    def _align_to_dap(
        self,
        means_cam: torch.Tensor,    # [N, 3] face-camera coords
        face_idx: int,
        dap_np: np.ndarray,          # [H, W] DAP depth
        H: int, W: int,
        scales_cam: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Align per-face Gaussian depths to DAP metric reference.

        Computes a single robust scale factor per face (log-median matching),
        then applies it to all positions and scales.
        Uses bilinear interpolation for DAP sampling.

        Returns:
            (means_cam_scaled, scale_factor) — the aligned means and the
            scale factor used, so the caller can do global consistency correction.
        """
        # Radial distance (euclidean norm) to match DAP's radial depth convention
        # DAP depth = distance along the ray, not the Z-component in camera space.
        # Using z_cam would bias scale at face edges where radial = z / cos(angle).
        r_cam = means_cam.norm(dim=-1).cpu().numpy()

        # Project Gaussian positions to ERP to sample DAP
        fwd = np.asarray(self.projector.face_directions[face_idx], dtype=np.float32)
        up_raw = np.asarray(self.projector.face_ups[face_idx], dtype=np.float32)
        right = np.cross(up_raw, fwd)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, fwd)
        up = up / (np.linalg.norm(up) + 1e-8)

        R = np.column_stack([right, up, fwd]).astype(np.float32)
        pos_cam_np = means_cam.cpu().numpy()  # [N, 3]
        pos_world_np = (R @ pos_cam_np.T).T   # [N, 3]

        # World coords to spherical (phi, theta) → ERP (row, col)
        x, y, z = pos_world_np[:, 0], pos_world_np[:, 1], pos_world_np[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2).clip(1e-8)
        phi = np.arccos(np.clip(y / r, -1, 1))              # colatitude [0, pi]
        theta = np.arctan2(-z, x) % (2 * np.pi)             # azimuth [0, 2pi]

        # Bilinear interpolation with horizontal wrap (ERP is 360° panorama)
        row_f = (phi / np.pi * (H - 1)).clip(0, H - 1.001)
        col_f = (1 - theta / (2 * np.pi)) * (W - 1)

        row0 = np.floor(row_f).astype(np.int32)
        row1 = np.minimum(row0 + 1, H - 1)
        col0 = np.floor(col_f).astype(np.int32) % W  # wrap horizontally
        col1 = (col0 + 1) % W                          # wrap to 0 at right edge

        fr = (row_f - row0).astype(np.float32)
        fc = (col_f - col0).astype(np.float32)

        dap_sampled = (
            dap_np[row0, col0] * (1 - fr) * (1 - fc) +
            dap_np[row1, col0] * fr * (1 - fc) +
            dap_np[row0, col1] * (1 - fr) * fc +
            dap_np[row1, col1] * fr * fc
        )

        # Log-median scale alignment (robust to outliers)
        valid = (r_cam > 0.1) & (dap_sampled > 0.1)
        if valid.sum() < 10:
            print(f"[SHARP-Direct]   DAP align face {face_idx}: only {valid.sum()} valid samples, skipping")
            return means_cam, 1.0  # not enough overlap, skip alignment

        log_ratio = np.log(dap_sampled[valid]) - np.log(r_cam[valid])
        scale_factor = float(np.exp(np.median(log_ratio)))
        scale_factor = np.clip(scale_factor, 0.05, 20.0)  # safety clamp
        sharp_median = float(np.median(r_cam[valid]))
        dap_median = float(np.median(dap_sampled[valid]))
        print(f"[SHARP-Direct]   DAP align face {face_idx}: scale={scale_factor:.3f} "
              f"(SHARP median radial={sharp_median:.1f}, DAP median={dap_median:.1f}, "
              f"valid={valid.sum():,}/{len(r_cam):,})")

        # Apply to positions and scales
        means_cam = means_cam * scale_factor
        scales_cam *= scale_factor  # in-place (caller sees the change)
        return means_cam, scale_factor

    def _overlap_scale_corrections(
        self,
        means: torch.Tensor,
        face_labels: torch.Tensor,
        all_face_forwards: List[np.ndarray],
        num_faces: int,
    ) -> np.ndarray:
        """Compute per-face scale corrections from overlap zone geometry.

        In the seam band near face boundaries, adjacent faces see the same
        scene content.  Comparing radial distances in these shared zones gives
        direct SHARP-to-SHARP scale ratios, which are more reliable than
        per-face DAP alignment (DAP can give wildly different scales for
        sky/ground faces).

        Inspired by Ruben Frosali's approach: use splat density in overlap
        regions to estimate relative scale corrections.

        Returns:
            np.ndarray [num_faces] — multiplicative corrections, mean=1.0
        """
        N = means.shape[0]
        if N < 100 or num_faces < 2:
            return np.ones(num_faces)

        dirs = F.normalize(means, dim=-1).cpu().numpy()
        labels_np = face_labels.cpu().numpy()
        dists = means.norm(dim=-1).cpu().numpy()

        # Find seam-zone Gaussians (near Voronoi boundaries)
        fwd_stack = np.stack(all_face_forwards, axis=0)  # [F, 3]
        cos_angles = dirs @ fwd_stack.T  # [N, F]
        top2_idx = np.argpartition(-cos_angles, 2, axis=1)[:, :2]
        cos_top2 = np.take_along_axis(cos_angles, top2_idx, axis=1)
        cos_1st = cos_top2.max(axis=1)
        cos_2nd = cos_top2.min(axis=1)
        margin = cos_1st - cos_2nd  # guaranteed >= 0
        seam_mask = margin < 0.15
        seam_idx = np.where(seam_mask)[0]

        if len(seam_idx) < 100:
            return np.ones(num_faces)

        seam_labels = labels_np[seam_idx]
        seam_dists = dists[seam_idx]
        seam_dirs = dirs[seam_idx]

        # Bin by direction on a coarse equirectangular grid
        x, y, z = seam_dirs[:, 0], seam_dirs[:, 1], seam_dirs[:, 2]
        phi = np.arccos(np.clip(y, -1, 1))
        theta = np.arctan2(-z, x) % (2 * np.pi)
        grid_res = 64
        row = (phi / np.pi * (grid_res - 1)).clip(0, grid_res - 1).astype(np.int32)
        col = (theta / (2 * np.pi) * (grid_res * 2 - 1)).clip(
            0, grid_res * 2 - 1
        ).astype(np.int32)
        cell_id = row * grid_res * 2 + col
        num_cells = grid_res * grid_res * 2

        # Per-face, per-cell distance accumulation (vectorized scatter-add)
        combined_key = seam_labels * num_cells + cell_id
        max_key = num_faces * num_cells

        dist_sum = np.zeros(max_key, dtype=np.float64)
        dist_count = np.zeros(max_key, dtype=np.int32)
        np.add.at(dist_sum, combined_key, seam_dists)
        np.add.at(dist_count, combined_key, 1)

        has_data = dist_count > 0
        dist_mean = np.zeros(max_key, dtype=np.float64)
        dist_mean[has_data] = dist_sum[has_data] / dist_count[has_data]

        dist_mean_2d = dist_mean.reshape(num_faces, num_cells)
        cell_count_2d = dist_count.reshape(num_faces, num_cells)

        # Find cells where 2+ faces contribute data
        multi_face = (cell_count_2d > 0).sum(axis=0) >= 2
        n_multi = multi_face.sum()
        if n_multi < 10:
            return np.ones(num_faces)

        # Global mean distance per cell (weighted by Gaussian count)
        total_count = cell_count_2d.sum(axis=0).astype(np.float64)
        total_dist = np.zeros(num_cells, dtype=np.float64)
        for fi in range(num_faces):
            total_dist += dist_sum.reshape(num_faces, num_cells)[fi]
        global_mean = np.zeros(num_cells, dtype=np.float64)
        has_any = total_count > 0
        global_mean[has_any] = total_dist[has_any] / total_count[has_any]

        # Per-face correction: median ratio of global_mean / face_mean
        corrections = np.ones(num_faces, dtype=np.float64)
        for fi in range(num_faces):
            fi_in_multi = (cell_count_2d[fi] > 0) & multi_face
            fi_cells = np.where(fi_in_multi)[0]
            if len(fi_cells) < 5:
                continue
            ratios = global_mean[fi_cells] / dist_mean_2d[fi, fi_cells]
            valid_ratios = ratios[(ratios > 0.01) & (ratios < 100)]
            if len(valid_ratios) < 3:
                continue
            corrections[fi] = float(np.exp(np.median(np.log(valid_ratios))))

        # Normalize: preserve overall scene scale (mean correction = 1.0)
        log_mean = np.mean(np.log(corrections))
        corrections /= np.exp(log_mean)

        return corrections

    def _smooth_seam_colors(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        face_labels: torch.Tensor,
        all_face_forwards: List[np.ndarray],
        smooth_strength: float = 0.3,
        grid_res: int = 256,
    ) -> torch.Tensor:
        """Smooth colors near face Voronoi boundaries to reduce seam visibility.

        Uses a fast equirectangular grid to bin Gaussians by direction, then
        blends seam-zone colors toward the cross-face average in each bin.
        Fully vectorized — no per-point Python loops.

        Args:
            means: [N, 3] world positions
            colors: [N, 3] RGB
            face_labels: [N] face index per Gaussian
            all_face_forwards: list of face forward vectors
            smooth_strength: blend factor toward cross-face mean (0-1)
            grid_res: resolution of the spherical binning grid
        """
        if smooth_strength <= 0 or means.shape[0] == 0:
            return colors

        N = means.shape[0]
        num_faces = len(all_face_forwards)
        device = colors.device

        # Work on CPU numpy for speed with large arrays
        dirs = F.normalize(means, dim=-1).cpu().numpy()
        labels_np = face_labels.cpu().numpy()
        colors_np = colors.cpu().numpy()

        # Identify seam-zone Gaussians: those near a Voronoi boundary
        fwd_stack = np.stack(all_face_forwards, axis=0)  # [F, 3]
        cos_angles = dirs @ fwd_stack.T                    # [N, F]

        # Top-2 face cosines per Gaussian (sort after argpartition — order not guaranteed)
        top2_idx = np.argpartition(-cos_angles, 2, axis=1)[:, :2]
        cos_top2 = np.take_along_axis(cos_angles, top2_idx, axis=1)
        cos_1st = cos_top2.max(axis=1)
        cos_2nd = cos_top2.min(axis=1)
        # Margin: how much closer the nearest face is vs second-nearest
        # Small margin = near a face boundary
        margin = cos_1st - cos_2nd  # guaranteed >= 0

        # Seam band: Gaussians where margin < threshold
        # For cubemap (90° between faces), typical margin at boundary ≈ 0
        # Use a tight band to limit candidates
        seam_threshold = 0.15
        seam_mask = margin < seam_threshold
        seam_indices = np.where(seam_mask)[0]

        if len(seam_indices) < 2:
            return colors

        print(f"[SHARP-Direct]   Seam band: {len(seam_indices):,} Gaussians ({100*len(seam_indices)/N:.1f}%)")

        # --- Grid-based averaging ---
        # Map seam Gaussians to an equirectangular grid, compute per-cell
        # average color for each face, then blend toward cross-face average.
        seam_dirs = dirs[seam_indices]
        seam_labels = labels_np[seam_indices]
        seam_colors = colors_np[seam_indices].copy()

        # Direction → (row, col) on equirectangular grid
        x, y, z = seam_dirs[:, 0], seam_dirs[:, 1], seam_dirs[:, 2]
        phi = np.arccos(np.clip(y, -1, 1))                  # [0, pi]
        theta = np.arctan2(-z, x) % (2 * np.pi)             # [0, 2pi]
        row = (phi / np.pi * (grid_res - 1)).clip(0, grid_res - 1).astype(np.int32)
        col = (theta / (2 * np.pi) * (grid_res * 2 - 1)).clip(0, grid_res * 2 - 1).astype(np.int32)
        cell_id = row * grid_res * 2 + col

        # Accumulate per-cell, per-face color sums and counts
        num_cells = grid_res * grid_res * 2
        # Use a combined key: cell_id * num_faces + face_label
        combined_key = cell_id * num_faces + seam_labels
        max_key = num_cells * num_faces

        # Vectorized scatter-add for color accumulation
        color_sum = np.zeros((max_key, 3), dtype=np.float64)
        color_count = np.zeros(max_key, dtype=np.int32)
        np.add.at(color_sum, combined_key, seam_colors)
        np.add.at(color_count, combined_key, 1)

        # For each seam Gaussian, compute the average color from OTHER faces
        # in the same cell
        # Total color in this cell (all faces)
        cell_total_sum = np.zeros((num_cells, 3), dtype=np.float64)
        cell_total_count = np.zeros(num_cells, dtype=np.int32)
        np.add.at(cell_total_sum, cell_id, seam_colors)
        np.add.at(cell_total_count, cell_id, 1)

        # Per-Gaussian: cross-face average = (cell_total - same_face) / (cell_total_count - same_face_count)
        same_key = combined_key
        same_sum = color_sum[same_key]        # [S, 3] sum of same-face colors in this cell
        same_count = color_count[same_key]    # [S]

        cross_sum = cell_total_sum[cell_id] - same_sum
        cross_count = cell_total_count[cell_id] - same_count

        # Only blend where cross-face neighbors exist
        has_cross = cross_count > 0
        if not has_cross.any():
            return colors

        cross_avg = np.zeros_like(seam_colors)
        cross_avg[has_cross] = cross_sum[has_cross] / cross_count[has_cross, np.newaxis]

        # Blend: stronger near boundary (small margin), weaker further away
        margin_seam = margin[seam_indices]
        # Normalize margin to [0, 1] within the seam band
        blend_weight = (1.0 - margin_seam / seam_threshold).clip(0, 1) * smooth_strength

        blended = seam_colors.copy()
        mask = has_cross
        blended[mask] = (
            (1 - blend_weight[mask, np.newaxis]) * seam_colors[mask]
            + blend_weight[mask, np.newaxis] * cross_avg[mask]
        )

        result = colors_np.copy()
        result[seam_indices] = np.clip(blended, 0, 1)
        return torch.from_numpy(result).to(device)
