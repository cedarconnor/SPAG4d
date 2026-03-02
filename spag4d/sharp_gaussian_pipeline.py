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


def _global_scale_alignment(
    all_means_world: List[torch.Tensor],
    all_scales: List[torch.Tensor],
    all_face_forwards: List[np.ndarray],
    face_fov: float,
    dap_depth_np: Optional[np.ndarray] = None,
    erp_H: int = 0,
    erp_W: int = 0,
    exclude_faces: Optional[set] = None,
    feature_ratios: Optional[List[Tuple[int, int, float, int, float]]] = None,
) -> List[torch.Tensor]:
    """Globally consistent face alignment via pairwise overlap ratios.

    The fundamental problem: per-face DAP alignment gives wildly different
    scale factors (e.g. 3x for ground, 20x for sky). Post-hoc corrections
    can't fix a 6x mismatch.

    Solution: measure pairwise scale ratios in overlap zones (SHARP-to-SHARP,
    which are reliable), then solve a global least-squares system for per-face
    scales. Finally, apply ONE global DAP scale to anchor the scene to metric.

    This replaces both the old per-face DAP alignment AND the old Procrustes.

    Args:
        all_means_world: list of [N_i, 3] world-frame positions per face
                         (already DAP-aligned individually — we'll undo that)
        all_scales: list of [N_i, 3] scale tensors per face
        all_face_forwards: face center direction vectors
        face_fov: face FOV in radians
        dap_depth_np: [H, W] DAP depth map for global metric anchoring
        erp_H, erp_W: ERP image dimensions

    Returns:
        all_means_world (modified in-place with globally consistent scales)
    """
    num_faces = len(all_means_world)
    if num_faces < 2:
        return all_means_world

    if exclude_faces is None:
        exclude_faces = set()

    device = all_means_world[0].device

    # --- Step 1: Find adjacent face pairs (skip excluded faces) ---
    fwd_stack = np.stack(all_face_forwards, axis=0)
    cos_matrix = fwd_stack @ fwd_stack.T
    adjacency_threshold = np.cos(face_fov * 0.8)

    pairs = []
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            if i in exclude_faces or j in exclude_faces:
                continue
            if cos_matrix[i, j] > adjacency_threshold:
                pairs.append((i, j))

    if not pairs:
        return all_means_world

    if exclude_faces:
        print(f"[SHARP-Direct]   Global alignment: {len(pairs)} pairs "
              f"(excluded faces: {sorted(exclude_faces)})")
    else:
        print(f"[SHARP-Direct]   Global alignment: {len(pairs)} adjacent face pairs")

    # Pre-compute unit directions
    all_dirs = []
    for i in range(num_faces):
        if all_means_world[i].shape[0] > 0:
            all_dirs.append(F.normalize(all_means_world[i], dim=-1))
        else:
            all_dirs.append(torch.zeros(0, 3, device=device))

    # --- Step 2: Measure pairwise scale ratios ---
    # Use SIFT feature-matched ratios as primary (3× confidence), then fall
    # back to NN angular matching for pairs without feature measurements.
    pairwise_log_ratios = []  # (i, j, log_ratio, n_matches, confidence)

    # Track which pairs have feature-based measurements
    feature_pairs = set()
    if feature_ratios:
        for fi, fj, lr, nm, conf in feature_ratios:
            # 3× confidence boost for geometrically verified matches
            pairwise_log_ratios.append((fi, fj, lr, nm, conf * 3.0))
            feature_pairs.add((min(fi, fj), max(fi, fj)))
        print(f"[SHARP-Direct]   {len(feature_ratios)} SIFT-based pairwise measurements")

    # NN angular matching as fallback for pairs without SIFT matches
    for fi, fj in pairs:
        if (min(fi, fj), max(fi, fj)) in feature_pairs:
            continue  # already have SIFT measurement for this pair

        if all_means_world[fi].shape[0] < 50 or all_means_world[fj].shape[0] < 50:
            continue

        dirs_i = all_dirs[fi]
        dirs_j = all_dirs[fj]

        fwd_j = torch.tensor(all_face_forwards[fj], dtype=torch.float32, device=device)
        fwd_i = torch.tensor(all_face_forwards[fi], dtype=torch.float32, device=device)

        cos_to_j = (dirs_i * fwd_j).sum(dim=-1)
        cos_to_i = (dirs_j * fwd_i).sum(dim=-1)

        overlap_thresh = np.cos(face_fov * 0.4)
        mask_i = cos_to_j > overlap_thresh
        mask_j = cos_to_i > overlap_thresh

        n_i = mask_i.sum().item()
        n_j = mask_j.sum().item()
        if n_i < 20 or n_j < 20:
            continue

        # Subsample for speed
        max_match = 5000
        if n_i > max_match:
            idx = torch.randperm(n_i, device=device)[:max_match]
            ov_dirs_i = dirs_i[mask_i][idx]
            ov_pos_i = all_means_world[fi][mask_i][idx]
        else:
            ov_dirs_i = dirs_i[mask_i]
            ov_pos_i = all_means_world[fi][mask_i]

        if n_j > max_match:
            idx = torch.randperm(n_j, device=device)[:max_match]
            ov_dirs_j = dirs_j[mask_j][idx]
            ov_pos_j = all_means_world[fj][mask_j][idx]
        else:
            ov_dirs_j = dirs_j[mask_j]
            ov_pos_j = all_means_world[fj][mask_j]

        # Nearest-neighbor matching in batches
        batch_size = 1000
        matched_r_i = []
        matched_r_j = []

        for start in range(0, ov_dirs_i.shape[0], batch_size):
            end = min(start + batch_size, ov_dirs_i.shape[0])
            cos_sim = ov_dirs_i[start:end] @ ov_dirs_j.T
            best_j = cos_sim.argmax(dim=1)
            best_cos = cos_sim[torch.arange(end - start, device=device), best_j]
            good = best_cos > 0.999
            if good.any():
                matched_r_i.append(ov_pos_i[start:end][good].norm(dim=-1))
                matched_r_j.append(ov_pos_j[best_j[good]].norm(dim=-1))

        if not matched_r_i:
            continue

        r_i = torch.cat(matched_r_i)
        r_j = torch.cat(matched_r_j)
        valid = (r_i > 0.1) & (r_j > 0.1)
        if valid.sum() < 10:
            continue

        log_ratio = torch.log(r_i[valid] / r_j[valid])
        median_lr = torch.median(log_ratio).item()
        # Confidence: inverse of IQR (tighter distribution = more reliable)
        iqr = float((torch.quantile(log_ratio, 0.75) - torch.quantile(log_ratio, 0.25)).item())
        confidence = 1.0 / max(iqr, 0.01)
        n_matches = valid.sum().item()

        pairwise_log_ratios.append((fi, fj, median_lr, n_matches, confidence))
        print(f"[SHARP-Direct]     NN pair ({fi},{fj}): {n_matches} matches, "
              f"ratio={np.exp(median_lr):.4f}, confidence={confidence:.1f}")

    if not pairwise_log_ratios:
        return all_means_world

    # --- Step 3: Solve global least-squares for per-face log-scales ---
    # System: for each pair (i,j) with measured log_ratio r_ij,
    #   log_scale_i - log_scale_j ≈ r_ij
    # This is an overdetermined system; solve via least-squares with
    # the constraint that sum(log_scale) = 0 (preserve overall scale).

    n_pairs = len(pairwise_log_ratios)
    A = np.zeros((n_pairs, num_faces), dtype=np.float64)
    b = np.zeros(n_pairs, dtype=np.float64)
    w = np.zeros(n_pairs, dtype=np.float64)

    for k, (fi, fj, lr, nm, conf) in enumerate(pairwise_log_ratios):
        A[k, fi] = 1.0
        A[k, fj] = -1.0
        b[k] = lr
        w[k] = conf * np.sqrt(nm)  # weight by confidence × sqrt(matches)

    # Weighted least-squares: minimize ||W(Ax - b)||²
    # Plus regularization toward zero (preserve original DAP scales)
    Aw = A * w[:, np.newaxis]
    bw = b * w

    # Very weak regularization — DAP normalization already handled the bulk.
    # The pairwise solver just needs to fix ~10-30% residual differences.
    reg_strength = 0.01 * np.mean(w)
    A_reg = np.eye(num_faces) * reg_strength
    b_reg = np.zeros(num_faces)

    A_full = np.vstack([Aw, A_reg])
    b_full = np.concatenate([bw, b_reg])

    # Solve via normal equations
    log_scales, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)

    # Normalize: mean log_scale = 0 (preserve overall scene scale)
    log_scales -= np.mean(log_scales)

    face_corrections = np.exp(log_scales)
    print(f"[SHARP-Direct]   Global scale corrections: "
          f"[{', '.join(f'{c:.3f}' for c in face_corrections)}]")

    # --- Step 4: Apply corrections ---
    for fi in range(num_faces):
        c = face_corrections[fi]
        if abs(c - 1.0) > 0.005 and all_means_world[fi].shape[0] > 0:
            all_means_world[fi] = all_means_world[fi] * c
            all_scales[fi] = all_scales[fi] * c

    # --- Step 5: Global DAP anchoring ---
    # Now all faces are mutually consistent. Apply ONE global scale to
    # match DAP metric depth (instead of per-face DAP alignment).
    if dap_depth_np is not None and erp_H > 0:
        all_log_ratios = []
        for fi in range(num_faces):
            pos = all_means_world[fi]
            if pos.shape[0] == 0:
                continue
            pos_np = pos.cpu().numpy()
            r_sharp = np.linalg.norm(pos_np, axis=-1)

            x, y, z = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]
            r = r_sharp.clip(1e-8)
            phi = np.arccos(np.clip(y / r, -1, 1))
            theta = np.arctan2(-z, x) % (2 * np.pi)

            row_f = (phi / np.pi * (erp_H - 1)).clip(0, erp_H - 1.001)
            col_f = (1 - theta / (2 * np.pi)) * (erp_W - 1)
            row0 = np.floor(row_f).astype(np.int32)
            row1 = np.minimum(row0 + 1, erp_H - 1)
            col0 = np.floor(col_f).astype(np.int32) % erp_W
            col1 = (col0 + 1) % erp_W
            fr = (row_f - row0).astype(np.float32)
            fc = (col_f - col0).astype(np.float32)

            dap_sampled = (
                dap_depth_np[row0, col0] * (1 - fr) * (1 - fc) +
                dap_depth_np[row1, col0] * fr * (1 - fc) +
                dap_depth_np[row0, col1] * (1 - fr) * fc +
                dap_depth_np[row1, col1] * fr * fc
            )

            valid = (r_sharp > 0.1) & (dap_sampled > 0.1)
            if valid.sum() > 100:
                lr = np.log(dap_sampled[valid]) - np.log(r_sharp[valid])
                all_log_ratios.append(lr)

        if all_log_ratios:
            combined = np.concatenate(all_log_ratios)
            global_dap_scale = float(np.exp(np.median(combined)))
            global_dap_scale = np.clip(global_dap_scale, 0.1, 10.0)
            print(f"[SHARP-Direct]   Global DAP metric scale: {global_dap_scale:.3f}")

            for fi in range(num_faces):
                if all_means_world[fi].shape[0] > 0:
                    all_means_world[fi] = all_means_world[fi] * global_dap_scale
                    all_scales[fi] = all_scales[fi] * global_dap_scale

    return all_means_world


def _feature_match_pairwise_ratios(
    faces_list,
    face_depth_maps: Dict[int, np.ndarray],
    face_scale_factors: List[float],
    cubemap_size: int,
    grid_size: int,
    face_fov: float,
    all_face_forwards: List[np.ndarray],
    exclude_faces: Optional[set] = None,
    min_matches: int = 10,
) -> List[Tuple[int, int, float, int, float]]:
    """Use SIFT feature-point matching to compute pairwise depth ratios.

    For each adjacent face pair, extracts SIFT keypoints from both face images,
    matches them with Lowe's ratio test + RANSAC, then compares SHARP depth at
    verified corresponding pixel locations.

    Returns list of (fi, fj, median_log_ratio, n_matches, confidence) — same
    format as _global_scale_alignment Step 2 pairwise measurements.
    """
    import cv2

    num_faces = len(faces_list)
    if exclude_faces is None:
        exclude_faces = set()

    # --- Find adjacent pairs (same logic as _global_scale_alignment) ---
    fwd_stack = np.stack(all_face_forwards, axis=0)
    cos_matrix = fwd_stack @ fwd_stack.T
    adjacency_threshold = np.cos(face_fov * 0.8)

    pairs = []
    for i in range(num_faces):
        for j in range(i + 1, num_faces):
            if i in exclude_faces or j in exclude_faces:
                continue
            if cos_matrix[i, j] > adjacency_threshold:
                pairs.append((i, j))

    if not pairs:
        return []

    # --- SIFT extraction per face (cached) ---
    sift = cv2.SIFT_create(nfeatures=2000)
    face_kp_desc = {}  # face_idx -> (keypoints, descriptors)

    needed_faces = set()
    for i, j in pairs:
        needed_faces.add(i)
        needed_faces.add(j)

    for fi in needed_faces:
        face_np = faces_list[fi]
        # Resize to grid_size×grid_size (768) to match depth map resolution
        face_resized = cv2.resize(face_np, (grid_size, grid_size))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        face_kp_desc[fi] = (kp, desc)

    # --- Match each pair ---
    bf = cv2.BFMatcher(cv2.NORM_L2)
    results = []

    for fi, fj in pairs:
        kp_i, desc_i = face_kp_desc[fi]
        kp_j, desc_j = face_kp_desc[fj]

        if desc_i is None or desc_j is None or len(kp_i) < 10 or len(kp_j) < 10:
            continue

        # KNN match + Lowe's ratio test
        raw_matches = bf.knnMatch(desc_i, desc_j, k=2)
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < min_matches:
            continue

        # RANSAC geometric filter
        pts_i = np.float32([kp_i[m.queryIdx].pt for m in good_matches])
        pts_j = np.float32([kp_j[m.trainIdx].pt for m in good_matches])

        _, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 5.0)
        if mask is None:
            continue

        inlier_mask = mask.ravel().astype(bool)
        pts_i = pts_i[inlier_mask]
        pts_j = pts_j[inlier_mask]

        if len(pts_i) < min_matches:
            continue

        # --- Depth lookup at matched keypoint locations ---
        depth_map_i = face_depth_maps.get(fi)
        depth_map_j = face_depth_maps.get(fj)
        if depth_map_i is None or depth_map_j is None:
            continue

        sf_i = face_scale_factors[fi] if fi < len(face_scale_factors) else 1.0
        sf_j = face_scale_factors[fj] if fj < len(face_scale_factors) else 1.0

        # OpenCV keypoint.pt = (x, y) = (col, row)
        rows_i = np.round(pts_i[:, 1]).astype(int).clip(0, grid_size - 1)
        cols_i = np.round(pts_i[:, 0]).astype(int).clip(0, grid_size - 1)
        rows_j = np.round(pts_j[:, 1]).astype(int).clip(0, grid_size - 1)
        cols_j = np.round(pts_j[:, 0]).astype(int).clip(0, grid_size - 1)

        d_i = depth_map_i[rows_i, cols_i] * sf_i
        d_j = depth_map_j[rows_j, cols_j] * sf_j

        valid = (d_i > 0.1) & (d_j > 0.1)
        if valid.sum() < min_matches:
            continue

        log_ratio = np.log(d_i[valid] / d_j[valid])
        median_lr = float(np.median(log_ratio))
        iqr = float(np.quantile(log_ratio, 0.75) - np.quantile(log_ratio, 0.25))
        confidence = 1.0 / max(iqr, 0.01)
        n_matches = int(valid.sum())

        results.append((fi, fj, median_lr, n_matches, confidence))
        print(f"[SHARP-Direct]     SIFT pair ({fi},{fj}): {n_matches} matches, "
              f"ratio={np.exp(median_lr):.4f}, confidence={confidence:.1f}")

    return results


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
        face_depth_maps = {}  # face_idx -> [grid_size, grid_size] front-layer depth

        grid_size = self.SHARP_INPUT_SIZE // 2  # 768

        for face_idx in range(num_faces):
            t_face = time.time()
            face_np = faces_list[face_idx]
            face_t = torch.from_numpy(face_np).float().to(self.device) / 255.0

            # --- Run SHARP ---
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: running SHARP inference...", flush=True)
            t0 = time.time()
            gaussians = self._run_sharp(face_t, f_px)
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: SHARP inference done in {time.time() - t0:.1f}s")

            # --- Store front-layer depth map for SIFT matching ---
            front_means_raw = gaussians.mean_vectors[0][:grid_size * grid_size]
            face_depth_maps[face_idx] = (
                front_means_raw.view(grid_size, grid_size, 3)
                .norm(dim=-1).cpu().numpy()
            )

            # --- Extract Gaussians (both layers) ---
            means_cam, scales_cam, quats_cam, colors, opacities = (
                self._unpack_gaussians(gaussians, grid_size)
            )
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: unpacked {means_cam.shape[0]:,} Gaussians (front+back)")
            # means_cam: [N, 3] in face-camera metric coords (right, up, forward)
            # quats_cam: [N, 4] WXYZ in face-camera frame

            del gaussians
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # --- Per-face DAP pre-alignment (initial scale estimate) ---
            # This gives each face a rough scale to match DAP. The global
            # solver will then fix inconsistencies between faces.
            t0 = time.time()
            means_cam, face_sf = self._align_to_dap(
                means_cam, face_idx, dap_np, H, W, scales_cam
            )
            face_scale_factors.append(face_sf)

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
                print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: 0 valid Gaussians")
                all_means.append(torch.zeros(0, 3, device=self.device))
                all_scales.append(torch.zeros(0, 3, device=self.device))
                all_quats.append(torch.zeros(0, 4, device=self.device))
                all_colors.append(torch.zeros(0, 3, device=self.device))
                all_opacities.append(torch.zeros(0, device=self.device))
                continue

            # --- Transform to world frame ---
            R_world, q_world, fwd = face_rotations[face_idx]
            # Position: p_world = R @ p_cam
            means_world = (R_world @ means_cam.T).T  # [N, 3]
            # Quaternion: compose face rotation with per-Gaussian rotation
            quats_world = _quat_multiply(q_world, quats_cam)
            # Scales are invariant to rotation (eigen-values don't change)

            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: {n_valid:,} valid Gaussians ({time.time() - t_face:.1f}s)")

            all_means.append(means_world)
            all_scales.append(scales_cam)
            all_quats.append(quats_world)
            all_colors.append(colors)
            all_opacities.append(opacities)

        # --- Clamp DAP outliers (fix sky/degenerate faces) ---
        # Per-face DAP alignment gives wildly different scales (e.g. 3x ground
        # vs 20x sky). Only clamp OUTLIER faces (>2x from median) — don't
        # normalize all faces, since legitimate depth differences exist.
        if len(face_scale_factors) > 1 and len(all_means) > 1:
            sf = np.array(face_scale_factors)
            # Robust median excluding clamped values
            non_clamped = sf[sf < 19.9]
            if len(non_clamped) >= 2:
                target_scale = float(np.median(non_clamped))
            else:
                target_scale = float(np.median(sf))

            print(f"[SHARP-Direct] DAP scale factors: "
                  f"[{', '.join(f'{s:.2f}' for s in sf)}]")
            print(f"[SHARP-Direct] Median DAP scale (non-clamped): {target_scale:.2f}")

            # Only correct faces that are >2x away from median
            for i in range(len(all_means)):
                if i >= len(sf):
                    continue
                ratio = sf[i] / target_scale
                if ratio > 2.0:
                    # Scale too large — clamp down to 2x median
                    c = (target_scale * 2.0) / sf[i]
                    all_means[i] = all_means[i] * c
                    all_scales[i] = all_scales[i] * c
                    print(f"[SHARP-Direct]   Face {i}: clamped from {sf[i]:.2f}x to "
                          f"{sf[i]*c:.2f}x (correction={c:.3f})")
                elif ratio < 0.5:
                    # Scale too small — clamp up to 0.5x median
                    c = (target_scale * 0.5) / sf[i]
                    all_means[i] = all_means[i] * c
                    all_scales[i] = all_scales[i] * c
                    print(f"[SHARP-Direct]   Face {i}: clamped from {sf[i]:.2f}x to "
                          f"{sf[i]*c:.2f}x (correction={c:.3f})")

        # --- Pairwise overlap alignment (fine-tune using SHARP-to-SHARP) ---
        # After DAP outlier clamping, remaining face-to-face differences are
        # typically 10-50%. The global solver fixes these using SHARP-to-SHARP
        # comparison in overlap zones. Exclude unreliable faces (sky/clamped)
        # since their contradictory overlap measurements poison the system.
        unreliable_faces = set()
        if len(face_scale_factors) > 1:
            sf = np.array(face_scale_factors)
            for i, s in enumerate(sf):
                if s >= 19.9:  # hit safety clamp = unreliable
                    unreliable_faces.add(i)

        # --- SIFT feature-point matching for pairwise depth ratios ---
        # Compute effective scale factors (accounting for DAP outlier clamping)
        effective_sf = list(face_scale_factors)
        if len(face_scale_factors) > 1 and len(all_means) > 1:
            sf = np.array(face_scale_factors)
            non_clamped = sf[sf < 19.9]
            target_scale = float(np.median(non_clamped)) if len(non_clamped) >= 2 else float(np.median(sf))
            for i in range(min(len(effective_sf), len(all_means))):
                ratio = sf[i] / target_scale
                if ratio > 2.0:
                    effective_sf[i] = target_scale * 2.0
                elif ratio < 0.5:
                    effective_sf[i] = target_scale * 0.5

        sift_ratios = None
        if len(all_means) > 1 and face_depth_maps:
            print(f"[SHARP-Direct] Running SIFT feature matching...")
            t0 = time.time()
            sift_ratios = _feature_match_pairwise_ratios(
                faces_list, face_depth_maps, effective_sf,
                self.cubemap_size, grid_size,
                self.projector.face_fov, all_face_forwards,
                exclude_faces=unreliable_faces,
            )
            print(f"[SHARP-Direct] SIFT matching done in {time.time() - t0:.1f}s "
                  f"({len(sift_ratios) if sift_ratios else 0} pairs)")

        if len(all_means) > 1:
            print(f"[SHARP-Direct] Running pairwise overlap alignment...")
            t0 = time.time()
            _global_scale_alignment(
                all_means, all_scales, all_face_forwards,
                self.projector.face_fov,
                dap_depth_np=dap_np, erp_H=H, erp_W=W,
                exclude_faces=unreliable_faces,
                feature_ratios=sift_ratios,
            )
            print(f"[SHARP-Direct] Pairwise alignment done in {time.time() - t0:.1f}s")

        # --- Apply Voronoi ownership + sky filter per face ---
        owned_means = []
        owned_scales = []
        owned_quats = []
        owned_colors = []
        owned_opacities = []
        all_face_labels = []

        for face_idx in range(len(all_means)):
            means_world = all_means[face_idx]
            scales_cam = all_scales[face_idx]
            quats_world = all_quats[face_idx]
            colors = all_colors[face_idx]
            opacities = all_opacities[face_idx]

            own = _nearest_face_mask(
                means_world, face_idx, all_face_forwards,
            )
            dist = means_world.norm(dim=-1)
            if sky_threshold > 0:
                keep = own & (dist < sky_threshold)
            else:
                keep = own

            owned_means.append(means_world[keep])
            owned_scales.append(scales_cam[keep])
            owned_quats.append(quats_world[keep])
            owned_colors.append(colors[keep])
            owned_opacities.append(opacities[keep])
            all_face_labels.append(
                torch.full((keep.sum().item(),), face_idx,
                           dtype=torch.long, device=means_world.device)
            )
            print(f"[SHARP-Direct] Face {face_idx+1}/{num_faces}: {means_world.shape[0]:,} -> {keep.sum().item():,} after ownership+sky")

        # Replace lists with owned versions
        all_means = owned_means
        all_scales = owned_scales
        all_quats = owned_quats
        all_colors = owned_colors
        all_opacities = owned_opacities

        # --- Merge faces ---
        if not all_means or all(m.shape[0] == 0 for m in all_means):
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

        # --- Spatially-varying overlap corrections ---
        # After global alignment, remaining local depth variation is handled
        # by a per-Gaussian correction field built from seam-zone comparisons.
        face_labels = torch.cat(all_face_labels, dim=0)
        if len(face_scale_factors) > 1:
            t0 = time.time()
            corrections = self._overlap_scale_corrections(
                means, face_labels, all_face_forwards, num_faces,
            )
            # corrections is now per-Gaussian [N] array
            corr_t = torch.from_numpy(corrections).float().to(means.device)
            adjusted = (corr_t - 1.0).abs() > 0.005
            if adjusted.any():
                means[adjusted] *= corr_t[adjusted].unsqueeze(-1)
                scales[adjusted] *= corr_t[adjusted].unsqueeze(-1)
            print(f"[SHARP-Direct] Overlap alignment done ({time.time() - t0:.1f}s)")

        # --- Soft geometric transition at seam boundaries ---
        if len(all_face_labels) > 1:
            print(f"[SHARP-Direct] Running seam geometry smoothing...", flush=True)
            t0 = time.time()
            means = self._smooth_seam_geometry(
                means, face_labels, all_face_forwards,
            )
            print(f"[SHARP-Direct] Seam geometry smoothing done in {time.time() - t0:.1f}s")

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
        """Compute spatially-varying per-Gaussian scale corrections from overlap zones.

        Instead of a single scalar per face, builds a correction field on a
        spherical grid by comparing radial depths where adjacent faces overlap.
        Each Gaussian gets a locally-interpolated correction factor, fixing
        spatially-varying depth misalignment (e.g. ground vs sky within one face).

        Returns:
            np.ndarray [N] — per-Gaussian multiplicative correction factors
        """
        N = means.shape[0]
        if N < 100 or num_faces < 2:
            return np.ones(N)

        dirs = F.normalize(means, dim=-1).cpu().numpy()
        labels_np = face_labels.cpu().numpy()
        dists = means.norm(dim=-1).cpu().numpy()

        # --- Identify seam-zone Gaussians ---
        fwd_stack = np.stack(all_face_forwards, axis=0)  # [F, 3]
        cos_angles = dirs @ fwd_stack.T  # [N, F]
        top2_idx = np.argpartition(-cos_angles, 2, axis=1)[:, :2]
        cos_top2 = np.take_along_axis(cos_angles, top2_idx, axis=1)
        cos_1st = cos_top2.max(axis=1)
        cos_2nd = cos_top2.min(axis=1)
        margin = cos_1st - cos_2nd
        seam_mask = margin < 0.20  # wider band for better spatial coverage
        seam_idx = np.where(seam_mask)[0]

        if len(seam_idx) < 100:
            return np.ones(N)

        seam_labels = labels_np[seam_idx]
        seam_dists = dists[seam_idx]
        seam_dirs = dirs[seam_idx]

        # --- Bin seam Gaussians on spherical grid ---
        grid_res = 128  # higher res for spatially-varying correction
        grid_cols = grid_res * 2
        num_cells = grid_res * grid_cols

        x, y, z = seam_dirs[:, 0], seam_dirs[:, 1], seam_dirs[:, 2]
        phi = np.arccos(np.clip(y, -1, 1))
        theta = np.arctan2(-z, x) % (2 * np.pi)
        row = (phi / np.pi * (grid_res - 1)).clip(0, grid_res - 1).astype(np.int32)
        col = (theta / (2 * np.pi) * (grid_cols - 1)).clip(0, grid_cols - 1).astype(np.int32)
        cell_id = row * grid_cols + col

        # Per-face, per-cell distance accumulation
        combined_key = seam_labels * num_cells + cell_id
        max_key = num_faces * num_cells

        dist_sum = np.zeros(max_key, dtype=np.float64)
        dist_count = np.zeros(max_key, dtype=np.int32)
        np.add.at(dist_sum, combined_key, seam_dists)
        np.add.at(dist_count, combined_key, 1)

        dist_mean_2d = np.zeros((num_faces, num_cells), dtype=np.float64)
        cell_count_2d = dist_count.reshape(num_faces, num_cells)
        has_data = cell_count_2d > 0
        dist_mean_2d[has_data] = dist_sum.reshape(num_faces, num_cells)[has_data] / cell_count_2d[has_data]

        # --- Build per-face correction grid ---
        # For each cell with 2+ faces, compute ratio of global mean to face mean
        faces_per_cell = (cell_count_2d > 0).sum(axis=0)
        multi_face = faces_per_cell >= 2

        # Global mean per cell (count-weighted average across faces)
        total_count = cell_count_2d.sum(axis=0).astype(np.float64)
        total_dist = dist_sum.reshape(num_faces, num_cells).sum(axis=0)
        global_mean = np.zeros(num_cells, dtype=np.float64)
        has_any = total_count > 0
        global_mean[has_any] = total_dist[has_any] / total_count[has_any]

        # Per-face correction grid: ratio at each cell
        correction_grid = np.ones((num_faces, grid_res, grid_cols), dtype=np.float64)
        has_correction = np.zeros((num_faces, grid_res, grid_cols), dtype=bool)

        for fi in range(num_faces):
            fi_cells = (cell_count_2d[fi] > 0) & multi_face
            cells = np.where(fi_cells)[0]
            if len(cells) < 3:
                continue
            ratios = global_mean[cells] / dist_mean_2d[fi, cells]
            valid = (ratios > 0.1) & (ratios < 10.0)
            cells = cells[valid]
            ratios = ratios[valid]
            if len(cells) < 3:
                continue
            cell_rows = cells // grid_cols
            cell_cols = cells % grid_cols
            correction_grid[fi, cell_rows, cell_cols] = ratios
            has_correction[fi, cell_rows, cell_cols] = True

        # --- Smooth the correction grids with Gaussian blur ---
        # This fills gaps and creates smooth spatial transitions
        from scipy.ndimage import gaussian_filter
        sigma = 4.0  # smoothing radius in grid cells

        for fi in range(num_faces):
            if not has_correction[fi].any():
                continue
            # Weighted smoothing: only spread from cells that have data
            weight = has_correction[fi].astype(np.float64)
            # Log-domain smoothing for multiplicative corrections
            log_corr = np.log(correction_grid[fi])
            log_corr[~has_correction[fi]] = 0.0

            smoothed_num = gaussian_filter(log_corr * weight, sigma=sigma, mode='wrap')
            smoothed_den = gaussian_filter(weight, sigma=sigma, mode='wrap')

            valid_smooth = smoothed_den > 0.01
            correction_grid[fi][valid_smooth] = np.exp(
                smoothed_num[valid_smooth] / smoothed_den[valid_smooth]
            )
            correction_grid[fi][~valid_smooth] = 1.0

        # --- Normalize correction grids (preserve scene scale) ---
        # Compute weighted mean correction across all faces and cells
        all_log_corr = []
        for fi in range(num_faces):
            fi_mask = labels_np == fi
            if fi_mask.sum() == 0:
                continue
            all_log_corr.append(np.log(correction_grid[fi]).mean())
        if all_log_corr:
            log_bias = np.mean(all_log_corr)
            for fi in range(num_faces):
                correction_grid[fi] = np.exp(np.log(correction_grid[fi]) - log_bias)

        # --- Sample correction for every Gaussian (not just seam) ---
        all_dirs = dirs  # [N, 3]
        ax, ay, az = all_dirs[:, 0], all_dirs[:, 1], all_dirs[:, 2]
        all_phi = np.arccos(np.clip(ay, -1, 1))
        all_theta = np.arctan2(-az, ax) % (2 * np.pi)
        all_row = (all_phi / np.pi * (grid_res - 1)).clip(0, grid_res - 1).astype(np.int32)
        all_col = (all_theta / (2 * np.pi) * (grid_cols - 1)).clip(0, grid_cols - 1).astype(np.int32)

        per_gaussian_correction = np.ones(N, dtype=np.float64)
        for fi in range(num_faces):
            fi_mask = labels_np == fi
            if fi_mask.sum() == 0:
                continue
            per_gaussian_correction[fi_mask] = correction_grid[fi, all_row[fi_mask], all_col[fi_mask]]

        # Safety clamp
        per_gaussian_correction = np.clip(per_gaussian_correction, 0.3, 3.0)

        n_adjusted = np.sum(np.abs(per_gaussian_correction - 1.0) > 0.005)
        print(f"[SHARP-Direct]   Spatially-varying correction: {n_adjusted:,}/{N:,} Gaussians adjusted")

        return per_gaussian_correction

    def _smooth_seam_geometry(
        self,
        means: torch.Tensor,
        face_labels: torch.Tensor,
        all_face_forwards: List[np.ndarray],
        smooth_strength: float = 0.8,
        grid_res: int = 64,
    ) -> torch.Tensor:
        """Smooth positions near face Voronoi boundaries to reduce geometric seams.

        For Gaussians in the seam band, computes the average radial depth from
        neighboring faces in the same angular bin, then blends the Gaussian's
        radial distance toward that average. The direction is preserved — only
        the depth changes. This eliminates "step" artifacts at face boundaries
        without causing ghosting (no opacity change, no doubled geometry).

        Args:
            means: [N, 3] world positions
            face_labels: [N] face index per Gaussian
            all_face_forwards: list of face forward vectors
            smooth_strength: blend factor toward cross-face depth (0-1)
            grid_res: resolution of the spherical binning grid
        """
        if smooth_strength <= 0 or means.shape[0] == 0:
            return means

        N = means.shape[0]
        num_faces = len(all_face_forwards)
        device = means.device

        dirs_np = F.normalize(means, dim=-1).cpu().numpy()
        labels_np = face_labels.cpu().numpy()
        dists_np = means.norm(dim=-1).cpu().numpy()

        # Find seam-zone Gaussians
        fwd_stack = np.stack(all_face_forwards, axis=0)
        cos_angles = dirs_np @ fwd_stack.T
        top2_idx = np.argpartition(-cos_angles, 2, axis=1)[:, :2]
        cos_top2 = np.take_along_axis(cos_angles, top2_idx, axis=1)
        margin = cos_top2.max(axis=1) - cos_top2.min(axis=1)

        seam_threshold = 0.25  # wider band for better cross-face coverage
        seam_mask = margin < seam_threshold
        seam_idx = np.where(seam_mask)[0]

        if len(seam_idx) < 10:
            return means

        # Bin seam Gaussians on ERP grid
        grid_cols = grid_res * 2
        num_cells = grid_res * grid_cols

        seam_dirs = dirs_np[seam_idx]
        seam_labels = labels_np[seam_idx]
        seam_dists = dists_np[seam_idx]

        x, y, z = seam_dirs[:, 0], seam_dirs[:, 1], seam_dirs[:, 2]
        phi = np.arccos(np.clip(y, -1, 1))
        theta = np.arctan2(-z, x) % (2 * np.pi)
        row = (phi / np.pi * (grid_res - 1)).clip(0, grid_res - 1).astype(np.int32)
        col = (theta / (2 * np.pi) * (grid_cols - 1)).clip(0, grid_cols - 1).astype(np.int32)
        cell_id = row * grid_cols + col

        # Per-cell, per-face distance accumulation
        # Layout: (num_faces, num_cells) so face_dist[fi, cell] works correctly
        combined_key = seam_labels * num_cells + cell_id
        max_key = num_faces * num_cells

        dist_sum = np.zeros(max_key, dtype=np.float64)
        dist_count = np.zeros(max_key, dtype=np.int32)
        np.add.at(dist_sum, combined_key, seam_dists)
        np.add.at(dist_count, combined_key, 1)

        # Total per cell (all faces)
        cell_total_sum = np.zeros(num_cells, dtype=np.float64)
        cell_total_count = np.zeros(num_cells, dtype=np.int32)
        np.add.at(cell_total_sum, cell_id, seam_dists)
        np.add.at(cell_total_count, cell_id, 1)

        # Cross-face average depth per seam Gaussian
        same_key = combined_key
        same_sum = dist_sum[same_key]
        same_count = dist_count[same_key]

        cross_sum = cell_total_sum[cell_id] - same_sum
        cross_count = cell_total_count[cell_id] - same_count

        has_cross = cross_count > 0
        if not has_cross.any():
            return means

        cross_avg_dist = np.zeros(len(seam_idx), dtype=np.float64)
        cross_avg_dist[has_cross] = cross_sum[has_cross] / cross_count[has_cross]

        # Build per-face correction field on spherical grid, then smooth
        # to cover Gaussians that lack direct cross-face data in their cell.
        from scipy.ndimage import gaussian_filter

        face_correction_ratio = np.ones((num_faces, grid_res, grid_cols), dtype=np.float64)
        has_ratio = np.zeros((num_faces, grid_res, grid_cols), dtype=bool)

        # Per-face mean depth per cell
        face_dist = dist_sum.reshape(num_faces, num_cells)
        face_count = dist_count.reshape(num_faces, num_cells)

        for fi in range(num_faces):
            fi_seam = seam_labels == fi
            if fi_seam.sum() == 0:
                continue
            fi_cells = cell_id[fi_seam]
            fi_dists = seam_dists[fi_seam]
            fi_cross_sum = cell_total_sum[fi_cells] - face_dist[fi, fi_cells].clip(0)
            fi_cross_cnt = cell_total_count[fi_cells] - face_count[fi, fi_cells].clip(0)
            fi_has_cross = fi_cross_cnt > 0
            if not fi_has_cross.any():
                continue
            fi_cross_avg = np.zeros(fi_seam.sum(), dtype=np.float64)
            fi_cross_avg[fi_has_cross] = fi_cross_sum[fi_has_cross] / fi_cross_cnt[fi_has_cross]

            # Compute ratio: cross-face avg / own depth
            fi_own_valid = (fi_dists > 0.1) & fi_has_cross & (fi_cross_avg > 0.1)
            if fi_own_valid.sum() == 0:
                continue
            ratios = fi_cross_avg[fi_own_valid] / fi_dists[fi_own_valid]
            ratio_cells = fi_cells[fi_own_valid]
            ratio_rows = ratio_cells // grid_cols
            ratio_cols = ratio_cells % grid_cols

            # Store ratios in grid (use median per cell for robustness)
            unique_cells = np.unique(ratio_cells)
            for uc in unique_cells:
                uc_mask = ratio_cells == uc
                r_median = np.median(ratios[uc_mask])
                if 0.2 < r_median < 5.0:
                    rr, rc = uc // grid_cols, uc % grid_cols
                    face_correction_ratio[fi, rr, rc] = r_median
                    has_ratio[fi, rr, rc] = True

        # Smooth the correction fields to fill gaps (key improvement)
        sigma_smooth = 5.0  # ~14° smoothing radius
        for fi in range(num_faces):
            if not has_ratio[fi].any():
                continue
            weight = has_ratio[fi].astype(np.float64)
            log_r = np.log(face_correction_ratio[fi])
            log_r[~has_ratio[fi]] = 0.0
            sm_num = gaussian_filter(log_r * weight, sigma=sigma_smooth, mode='wrap')
            sm_den = gaussian_filter(weight, sigma=sigma_smooth, mode='wrap')
            valid = sm_den > 0.01
            face_correction_ratio[fi][valid] = np.exp(sm_num[valid] / sm_den[valid])
            face_correction_ratio[fi][~valid] = 1.0

        # Sample smoothed correction for each seam Gaussian
        seam_corrections = np.ones(len(seam_idx), dtype=np.float64)
        for fi in range(num_faces):
            fi_mask = seam_labels == fi
            if fi_mask.sum() == 0:
                continue
            fi_rows = row[fi_mask]
            fi_cols = col[fi_mask]
            seam_corrections[fi_mask] = face_correction_ratio[fi, fi_rows, fi_cols]

        # Blend weight: stronger near boundary (small margin)
        margin_seam = margin[seam_idx]
        blend = (1.0 - margin_seam / seam_threshold).clip(0, 1) * smooth_strength

        # Compute new radial distance
        target_dist = seam_dists * seam_corrections
        new_dist = (1 - blend) * seam_dists + blend * target_dist

        # Apply: scale position along its direction
        scale_ratio = np.ones(len(seam_idx), dtype=np.float64)
        nonzero = seam_dists > 0.01
        scale_ratio[nonzero] = new_dist[nonzero] / seam_dists[nonzero]

        # Safety clamp
        scale_ratio = np.clip(scale_ratio, 0.3, 3.0)

        # Build full-size correction array
        result = means.clone()
        seam_idx_t = torch.from_numpy(seam_idx).long().to(device)
        ratio_t = torch.from_numpy(scale_ratio).float().to(device)

        adjusted_mask = (ratio_t - 1.0).abs() > 0.001
        if adjusted_mask.any():
            idx = seam_idx_t[adjusted_mask]
            r = ratio_t[adjusted_mask].unsqueeze(-1)
            result[idx] = result[idx] * r

        n_adjusted = adjusted_mask.sum().item()
        if n_adjusted > 0:
            avg_shift = np.abs(scale_ratio[adjusted_mask.cpu().numpy()] - 1.0).mean() * 100
        else:
            avg_shift = 0.0
        print(f"[SHARP-Direct]   Seam geometry: {n_adjusted:,}/{len(seam_idx):,} seam Gaussians adjusted "
              f"(avg shift: {avg_shift:.1f}%)")

        return result

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
