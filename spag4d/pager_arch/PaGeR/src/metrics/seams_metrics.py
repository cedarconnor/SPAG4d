import torch


def erp_seam_jumps(depth_erp: torch.Tensor, face_map: torch.Tensor):
    """Per-pixel |Δdepth| across cube-face seams on ERP, plus a unique edge_id per cube edge.

    Computed on ERP (not cube) because both sides of a seam sample the same (lat, lon).
    """
    if depth_erp.ndim == 4:
        d = depth_erp[0, 0]
    elif depth_erp.ndim == 3:
        d = depth_erp[0]
    else:
        d = depth_erp

    # Horizontal: ERP wraps in longitude, so include col W-1 ↔ col 0.
    h_diff = (d - torch.roll(d, shifts=-1, dims=1)).abs()
    h_a = face_map
    h_b = torch.roll(face_map, shifts=-1, dims=1)
    h_mask = h_a != h_b

    # Vertical: no wrap at the poles.
    v_diff = (d[1:] - d[:-1]).abs()
    v_a = face_map[:-1]
    v_b = face_map[1:]
    v_mask = v_a != v_b

    fa = torch.cat([h_a[h_mask], v_a[v_mask]]).long()
    fb = torch.cat([h_b[h_mask], v_b[v_mask]]).long()
    jumps = torch.cat([h_diff[h_mask], v_diff[v_mask]])
    edge_id = torch.minimum(fa, fb) * 6 + torch.maximum(fa, fb)
    return jumps, edge_id


def per_edge_jump_stats(jumps: torch.Tensor, edge_id: torch.Tensor,
                        pixel_threshold: float = None):
    """Per-cube-edge mean and max (12,). If pixel_threshold is given, also returns per-edge density."""
    # Edge ids live in [0, 36); only 12 are populated.
    sums = torch.zeros(36, device=jumps.device, dtype=jumps.dtype)
    counts = torch.zeros(36, device=jumps.device, dtype=jumps.dtype)
    maxes = torch.full((36,), float("-inf"), device=jumps.device, dtype=jumps.dtype)
    sums.index_add_(0, edge_id, jumps)
    counts.index_add_(0, edge_id, torch.ones_like(jumps))
    maxes.scatter_reduce_(0, edge_id, jumps, reduce="amax", include_self=True)
    valid = counts > 0
    means = sums[valid] / counts[valid]
    out_maxes = maxes[valid]
    if pixel_threshold is None:
        return means, out_maxes
    exceeds = torch.zeros(36, device=jumps.device, dtype=jumps.dtype)
    exceeds.index_add_(0, edge_id, (jumps > pixel_threshold).to(jumps.dtype))
    densities = exceeds[valid] / counts[valid]
    return means, out_maxes, densities


def equirect_facetype(
    h: int,
    w: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """ERP pixel → cubemap face index (0-5)."""
    tp = (
        torch.arange(4, device=device)
        .repeat_interleave(w // 4)
        .unsqueeze(0)
        .repeat(h, 1)
    )
    tp = torch.roll(tp, shifts=3 * (w // 8), dims=1)

    mask = torch.zeros((h, w // 4), dtype=torch.bool, device=device)
    idx = torch.linspace(-torch.pi, torch.pi, w // 4, device=device, dtype=dtype) / 4
    idx = torch.round(h / 2 - torch.atan(torch.cos(idx)) * h / torch.pi).to(torch.long)
    for i, j in enumerate(idx):
        mask[:j, i] = True
    mask = torch.roll(torch.cat([mask] * 4, dim=1), shifts=3 * (w // 8), dims=1)

    tp[mask] = 4
    tp[torch.flip(mask, [0])] = 5
    return tp


def minmax_normalize(depth: torch.Tensor, mask: torch.Tensor = None,
                     eps: float = 1e-8, percentile: float = 0.0) -> torch.Tensor:
    """Per-sample min-max → [0, 1]. percentile > 0 clips to that quantile range first (outlier-robust)."""
    B = depth.shape[0]
    out = torch.empty_like(depth)
    for i in range(B):
        flat = depth[i].reshape(-1)
        valid = flat[mask[i].reshape(-1).bool()] if mask is not None else flat
        if valid.numel() == 0:
            out[i] = depth[i]
            continue
        if percentile > 0.0:
            qs = torch.tensor([percentile, 1.0 - percentile],
                              device=valid.device, dtype=valid.dtype)
            lo, hi = torch.quantile(valid, qs)
        else:
            lo, hi = valid.min(), valid.max()
        scale = (hi - lo).clamp_min(eps)
        out[i] = ((depth[i] - lo) / scale).clamp(0.0, 1.0)
    return out


class MetricTracker:
    """Accumulates seam-quality metrics (defect_density, prevalence, severity) across samples."""

    def __init__(self, tracked_metrics,
                 pixel_jump_thresh: float = 0.02,
                 edge_mean_jump_thresh: float = 0.015,
                 min_broken_pixel_frac: float = 0.1,
                 normalize: bool = True, normalize_percentile: float = 0.01):
        self.tracked_metrics = tracked_metrics
        self.pixel_jump_thresh = pixel_jump_thresh
        self.edge_mean_jump_thresh = edge_mean_jump_thresh
        self.min_broken_pixel_frac = min_broken_pixel_frac
        self.normalize = normalize
        self.normalize_percentile = normalize_percentile
        self.metrics_sum = {metric: 0.0 for metric in tracked_metrics}

        self._face_map = None
        self._needs_edge_density = "seam_prevalence" in tracked_metrics

    def _ensure_face_map(self, H, W, device, dtype):
        if self._face_map is None or self._face_map.shape != (H, W):
            self._face_map = equirect_facetype(H, W, device=device, dtype=dtype)

    def seam_defect_density(self, ctx) -> float:
        # Fraction of seam pixels with |Δdepth| > pixel_jump_thresh.
        return (ctx["jumps"] > self.pixel_jump_thresh).float().mean().item()

    def seam_severity(self, ctx) -> float:
        # Fraction of the 12 edges whose mean |Δdepth| > edge_mean_jump_thresh.
        return (ctx["edge_means"] > self.edge_mean_jump_thresh).float().mean().item()

    def seam_prevalence(self, ctx) -> float:
        # Fraction of the 12 edges with enough broken pixels to count as a seam.
        return (ctx["edge_densities"] > self.min_broken_pixel_frac).float().mean().item()

    def update(self, depth_pred: torch.Tensor):
        if depth_pred.ndim == 3:
            depth_pred = depth_pred.unsqueeze(0)
        elif depth_pred.ndim == 2:
            depth_pred = depth_pred.unsqueeze(0).unsqueeze(0)

        H, W = depth_pred.shape[-2], depth_pred.shape[-1]

        if self.normalize:
            depth_pred = minmax_normalize(depth_pred, percentile=self.normalize_percentile)

        self._ensure_face_map(H, W, depth_pred.device, depth_pred.dtype)
        jumps, edge_id = erp_seam_jumps(depth_pred, self._face_map)
        ctx = {"depth": depth_pred, "jumps": jumps}
        if self._needs_edge_density:
            edge_means, _, edge_densities = per_edge_jump_stats(
                jumps, edge_id, pixel_threshold=self.pixel_jump_thresh
            )
            ctx["edge_means"] = edge_means
            ctx["edge_densities"] = edge_densities
        else:
            edge_means, _ = per_edge_jump_stats(jumps, edge_id)
            ctx["edge_means"] = edge_means

        for metric in self.tracked_metrics:
            value = getattr(self, metric)(ctx)
            self.metrics_sum[metric] += value

    def calculate_final(self, num_samples):
        final_metrics = {}
        for metric in self.tracked_metrics:
            final_metrics[metric] = self.metrics_sum[metric] / max(num_samples, 1)
        return final_metrics
