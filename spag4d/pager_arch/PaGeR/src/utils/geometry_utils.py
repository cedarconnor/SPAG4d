import math
import torch
import numpy as np
import trimesh
from pytorch360convert import e2c, c2e, e2p


# (h_deg, v_deg) for [F, R, B, L, U, D] — matches e2c stack at fov=90.
_CUBE_FACE_ANGLES = (
    (0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (-90.0, 0.0), (0.0, 90.0), (0.0, -90.0),
)


def _fov_crop_size(face_w: int, fov_deg: float) -> int:
    if fov_deg <= 90.0:
        return face_w
    return int(round(face_w / math.tan(math.radians(fov_deg / 2.0))))


def _e2p_stack(erp_tensor, face_w, fov_deg, mode):
    faces = [
        e2p(erp_tensor, fov_deg=fov_deg, h_deg=h, v_deg=v,
            out_hw=(face_w, face_w), mode=mode)
        for h, v in _CUBE_FACE_ANGLES
    ]
    return torch.stack(faces, dim=0)


def _crop_overlap(cube_stack, face_w, fov_deg):
    # Strip the FOV>90 overlap so c2e (which assumes 90° faces) gets a gapless tiling.
    crop = _fov_crop_size(face_w, fov_deg)
    if crop == face_w:
        return cube_stack
    start = (face_w - crop) // 2
    end = start + crop
    return cube_stack[..., start:end, start:end]


def erp_to_cubemap(erp_tensor, face_w=504, fov=90.0, cube_format="stack", mode="bilinear", **kwargs):
    """ERP → cubemap stack. fov=90 uses fast e2c; fov>90 uses per-face e2p (overlapping faces)."""
    if fov == 90.0:
        if erp_tensor.ndim == 3:
            return e2c(erp_tensor, face_w=face_w, cube_format=cube_format, mode=mode, **kwargs)
        if erp_tensor.ndim == 4:
            return torch.stack([
                e2c(erp_tensor[b], face_w=face_w, cube_format=cube_format, mode=mode, **kwargs)
                for b in range(erp_tensor.shape[0])
            ], dim=0)
        raise ValueError(
            f"Unsupported erp_tensor shape {tuple(erp_tensor.shape)}. Expected (C, H, W) or (B, C, H, W)."
        )

    if cube_format != "stack":
        raise NotImplementedError(
            f"erp_to_cubemap with fov != 90 only supports cube_format='stack', got {cube_format!r}"
        )
    if erp_tensor.ndim == 3:
        return _e2p_stack(erp_tensor, face_w, fov, mode)
    if erp_tensor.ndim == 4:
        return torch.stack([
            _e2p_stack(erp_tensor[b], face_w, fov, mode)
            for b in range(erp_tensor.shape[0])
        ], dim=0)
    raise ValueError(
        f"Unsupported erp_tensor shape {tuple(erp_tensor.shape)}. Expected (C, H, W) or (B, C, H, W)."
    )


def cubemap_to_erp(cube_tensor, erp_h=1024, erp_w=2048, fov=90.0, cube_format="stack", mode="bilinear", **kwargs):
    """Cubemap → ERP. For fov>90, faces are center-cropped to 90° before c2e stitching."""
    if fov != 90.0:
        if cube_format != "stack":
            raise NotImplementedError(
                f"cubemap_to_erp with fov != 90 only supports cube_format='stack', got {cube_format!r}"
            )
        face_w = cube_tensor.shape[-1]
        cube_tensor = _crop_overlap(cube_tensor, face_w, fov)

    if cube_tensor.ndim == 4:
        if cube_tensor.shape[0] != 6:
            raise ValueError(
                f"Expected cube_tensor shape (6, C, h, w) for 4D input, got {tuple(cube_tensor.shape)}"
            )
        return c2e(cube_tensor, h=erp_h, w=erp_w, cube_format=cube_format, mode=mode, **kwargs)

    if cube_tensor.ndim == 5:
        if cube_tensor.shape[1] != 6:
            raise ValueError(
                f"Expected cube_tensor shape (B, 6, C, h, w) for 5D input, got {tuple(cube_tensor.shape)}"
            )
        erp_batch = []
        for b in range(cube_tensor.shape[0]):
            erp_batch.append(
                c2e(cube_tensor[b], h=erp_h, w=erp_w, cube_format=cube_format, mode=mode, **kwargs)
            )
        return torch.stack(erp_batch, dim=0)

    raise ValueError(
        f"Unsupported cube_tensor shape {tuple(cube_tensor.shape)}. Expected (6, C, h, w) or (B, 6, C, h, w)."
    )

def compute_scale_and_shift(pred_g, targ_g, mask_g=None, weights=None, eps=0.0, fit_shift=True):
    if mask_g is None:
        mask_g = torch.ones_like(pred_g, dtype=torch.bool)

    # contiguous() before reshape: avoids a view that an in-place op could corrupt in autograd.
    B = pred_g.shape[0]
    pred_g  = pred_g.contiguous().reshape(B, -1)
    targ_g  = targ_g.contiguous().reshape(B, -1)
    mask_g  = mask_g.contiguous().reshape(B, -1).to(dtype=pred_g.dtype)
    if weights is not None:
        weights = weights.contiguous().reshape(B, -1)

    mask_w = mask_g * weights if weights is not None else mask_g

    a_00 = torch.sum(mask_w * pred_g * pred_g, dim=1)
    a_01 = torch.sum(mask_w * pred_g,          dim=1)
    a_11 = torch.sum(mask_w,                   dim=1)
    b_0  = torch.sum(mask_w * pred_g * targ_g, dim=1)
    b_1  = torch.sum(mask_w * targ_g,          dim=1)

    if fit_shift:
        det = a_00 * a_11 - a_01 * a_01 + eps
        scale = torch.zeros_like(b_0)
        shift = torch.zeros_like(b_1)
        valid = det > 0
        scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
        return scale, shift
    else:
        scale = b_0 / (a_00 + eps)
        return scale, torch.zeros_like(scale)


def unit_normals(n, eps = 1e-6):
    assert n.dim() >= 3 and n.size(-3) == 3, "normals must have channel=3 at dim -3"
    denom = torch.clamp(torch.linalg.norm(n, dim=-3, keepdim=True), min=eps)
    return n / denom


def z_depth_to_euclidean(z_depth, intrinsics):
    squeeze_batch = False
    if z_depth.ndim == 4:
        if z_depth.shape[1] != 1:
            raise ValueError(
                f"Expected z_depth shape (S, 1, H, W) for 4D input, got {tuple(z_depth.shape)}"
            )
        z_depth = z_depth.unsqueeze(0)
        squeeze_batch = True
    elif z_depth.ndim == 5:
        if z_depth.shape[2] != 1:
            raise ValueError(
                f"Expected z_depth shape (B, S, 1, H, W) for 5D input, got {tuple(z_depth.shape)}"
            )
    else:
        raise ValueError(
            f"Unsupported z_depth shape {tuple(z_depth.shape)}. Expected (6, 1, H, W) or (B, 6, 1, H, W)."
        )

    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    _, _, _, H, W = z_depth.shape
    u = torch.arange(W, device=z_depth.device, dtype=z_depth.dtype)
    v = torch.arange(H, device=z_depth.device, dtype=z_depth.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    x_norm = (uu - cx) / fx
    y_norm = (vv - cy) / fy
    ray_length_multiplier = torch.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
    ray_length_multiplier = ray_length_multiplier.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    out = z_depth * ray_length_multiplier
    return out.squeeze(0) if squeeze_batch else out


def remove_isolated_clusters_3d(points: np.ndarray,
                                 max_cluster_size: int = 500,
                                 connect_factor: float = 0.05,
                                 isolation_factor: float = 0.2,
                                 far_percentile: float = 90.0) -> np.ndarray:
    """Drop small far clusters isolated from the main cloud; thresholds scale with r_far.

    Among points past the far_percentile, components are formed within
    connect_factor*r_far; components <= max_cluster_size whose closest near-cloud
    distance exceeds isolation_factor*r_far are removed. Returns (N,) keep mask.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    N = len(points)
    inlier = np.ones(N, dtype=bool)

    r = np.linalg.norm(points, axis=1)
    r_far = np.percentile(r, far_percentile)

    connect_radius   = connect_factor   * r_far
    isolation_radius = isolation_factor * r_far

    far_mask  = r > r_far
    near_mask = ~far_mask

    far_points  = points[far_mask]
    near_points = points[near_mask]
    far_idx     = np.where(far_mask)[0]

    if len(far_points) < 2 or len(near_points) == 0:
        return inlier

    far_tree = cKDTree(far_points)
    pairs = far_tree.query_pairs(connect_radius, output_type='ndarray')

    N_far = len(far_points)
    if len(pairs) > 0:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adj  = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(N_far, N_far))
    else:
        adj = csr_matrix((N_far, N_far))

    _, labels    = connected_components(adj, directed=False)
    comp_sizes   = np.bincount(labels)
    small_comp_ids = np.where(comp_sizes <= max_cluster_size)[0]

    if len(small_comp_ids) == 0:
        return inlier

    near_tree = cKDTree(near_points)

    for comp_id in small_comp_ids:
        comp_local = labels == comp_id
        comp_pts   = far_points[comp_local]
        dists, _ = near_tree.query(comp_pts, k=1, workers=-1)
        if dists.min() > isolation_radius:
            inlier[far_idx[comp_local]] = False

    return inlier


def compute_edge_mask(depth, abs_thresh = 0.1, rel_thresh = 0.1):
    assert depth.ndim == 2
    depth = depth.astype(np.float32, copy=False)

    valid = depth > 0
    eps = 1e-6

    edge = np.zeros_like(valid, dtype=bool)

    d1 = depth[:, :-1]
    d2 = depth[:, 1:]
    v_pair = valid[:, :-1] & valid[:, 1:]

    diff = np.abs(d1 - d2)
    rel = diff / (np.minimum(d1, d2) + eps)

    edge_pair = v_pair & (diff > abs_thresh) & (rel > rel_thresh)

    edge[:, :-1] |= edge_pair
    edge[:, 1:] |= edge_pair

    d1 = depth[:-1, :]
    d2 = depth[1:, :]
    v_pair = valid[:-1, :] & valid[1:, :]

    diff = np.abs(d1 - d2)
    rel = diff / (np.minimum(d1, d2) + eps)

    edge_pair = v_pair & (diff > abs_thresh) & (rel > rel_thresh)

    edge[:-1, :] |= edge_pair
    edge[1:, :]  |= edge_pair

    keep = valid & (~edge)
    return keep


_ERP_DIRS_CACHE: dict = {}


def _get_erp_dirs(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached (3, H, W) unit ray directions: x=cos(phi)cos(theta), y=sin(phi), z=cos(phi)sin(theta)."""
    key = (H, W, device, dtype)
    if key not in _ERP_DIRS_CACHE:
        u = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        v = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
        theta = u * (2.0 * torch.pi) - torch.pi
        phi   = (0.5 - v) * torch.pi

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_the = torch.cos(theta)
        sin_the = torch.sin(theta)

        dir_x = cos_phi[:, None] * cos_the[None, :]
        dir_y = sin_phi[:, None].expand(H, W)
        dir_z = cos_phi[:, None] * sin_the[None, :]

        _ERP_DIRS_CACHE[key] = torch.stack([dir_x, dir_y, dir_z], dim=0)
    return _ERP_DIRS_CACHE[key]


def erp_to_pointcloud(depth: torch.Tensor) -> torch.Tensor:
    """Differentiable ERP depth → XYZ. Accepts (H, W) | (B, H, W) | (B, 1, H, W)."""
    ndim = depth.dim()
    if ndim == 2:
        H, W = depth.shape
        dirs = _get_erp_dirs(H, W, depth.device, depth.dtype)
        return depth.unsqueeze(0) * dirs
    if ndim == 3:
        depth = depth.unsqueeze(1)
    B, _, H, W = depth.shape
    dirs = _get_erp_dirs(H, W, depth.device, depth.dtype)
    return depth * dirs.unsqueeze(0)


def get_cubemap_intrinsics_extrinsics(image_size=512, fov=90.0):
    """6-face cubemap cameras → (extrinsics (6,4,4), intrinsics (6,3,3)) on CPU float32."""
    S = float(image_size)
    f = S / (2.0 * math.tan(math.radians(fov / 2.0)))
    cx, cy = S / 2.0, S / 2.0

    K = torch.tensor([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    intrinsics = K.unsqueeze(0).expand(6, -1, -1).clone()

    face_configs = [
        (0, 0), (-90, 0), (180, 0), (90, 0), (0, -90), (0, 90),  # F, R, B, L, U, D
    ]

    face_configs = torch.tensor(face_configs, dtype=torch.float32)
    y = torch.deg2rad(face_configs[:, 0])
    p = torch.deg2rad(face_configs[:, 1])

    cy_r = torch.cos(y)
    sy = torch.sin(y)
    cp = torch.cos(p)
    sp = torch.sin(p)

    Ry = torch.zeros((6, 3, 3), dtype=torch.float32)
    Ry[:, 0, 0] = cy_r
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy_r

    Rx = torch.zeros((6, 3, 3), dtype=torch.float32)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cp
    Rx[:, 1, 2] = -sp
    Rx[:, 2, 1] = sp
    Rx[:, 2, 2] = cp

    R = torch.bmm(Rx, Ry)

    extrinsics = torch.zeros((6, 4, 4), dtype=torch.float32)
    extrinsics[:, :3, :3] = R
    extrinsics[:, 3, 3] = 1.0

    return extrinsics, intrinsics


def erp_to_point_cloud_glb(rgb, depth, mask=None, export_path=None,
                            remove_isolated_clusters: bool = True,
                            cluster_max_size: int = 500,
                            cluster_connect_factor: float = 0.05,
                            cluster_isolation_factor: float = 0.2,
                            cluster_far_percentile: float = 90.0):
    """Project ERP rgb+depth to a GLB; optionally strip isolated far clusters."""
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().float().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    depth = depth.astype(np.float32, copy=False)
    H, W  = depth.shape

    xyz_np = erp_to_pointcloud(torch.from_numpy(depth)).permute(1, 2, 0).numpy()

    keep = depth > 0
    if mask is not None:
        keep = keep & np.asarray(mask, dtype=bool)

    points = xyz_np[keep]
    colors = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[keep]

    if remove_isolated_clusters and len(points) > 0:
        inlier = remove_isolated_clusters_3d(
            points,
            max_cluster_size=cluster_max_size,
            connect_factor=cluster_connect_factor,
            isolation_factor=cluster_isolation_factor,
            far_percentile=cluster_far_percentile,
        )
        points = points[inlier]
        colors = colors[inlier]

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))
    scene.export(export_path)
    return scene
