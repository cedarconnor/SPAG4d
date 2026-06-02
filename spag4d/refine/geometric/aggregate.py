# spag4d/refine/geometric/aggregate.py
"""Voxel downsampling of surviving hole candidates."""
from dataclasses import dataclass

import numpy as np


@dataclass
class AggregatedCandidates:
    positions: np.ndarray   # (M, 3) voxel centroids
    colors: np.ndarray      # (M, 3) averaged RGB [0, 1]
    voxel_size: float


    @property
    def num_voxels(self) -> int:
        return len(self.positions)


def aggregate_candidates(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    max_gaussians: int = 2_000_000,
) -> AggregatedCandidates:
    """Voxel-downsample candidate points, averaging colors within each voxel.

    Retries with 1.5x voxel_size if output exceeds max_gaussians.

    Args:
        points: (K, 3) float32 world-space points.
        colors: (K, 3) float32 RGB colors [0, 1].
        voxel_size: initial voxel size.
        max_gaussians: upper bound on output count.

    Returns:
        AggregatedCandidates with centroids and averaged colors.
    """
    try:
        import open3d as o3d
        _aggregate_fn = _aggregate_open3d
    except ImportError:
        _aggregate_fn = _aggregate_numpy

    for _ in range(20):
        result = _aggregate_fn(points, colors, voxel_size)
        if result.num_voxels <= max_gaussians:
            return result
        voxel_size *= 1.5

    return result


def _aggregate_open3d(points, colors, voxel_size) -> AggregatedCandidates:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    out_pts = np.asarray(down.points, dtype=np.float32)
    out_col = np.asarray(down.colors, dtype=np.float32)
    return AggregatedCandidates(positions=out_pts, colors=out_col, voxel_size=voxel_size)


def _aggregate_numpy(points, colors, voxel_size) -> AggregatedCandidates:
    """Fallback voxel downsample without open3d."""
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    keys = [tuple(r) for r in voxel_idx]
    unique_keys = list(dict.fromkeys(keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    assignment = np.array([key_to_idx[k] for k in keys], dtype=np.int32)

    M = len(unique_keys)
    out_pts = np.zeros((M, 3), dtype=np.float32)
    out_col = np.zeros((M, 3), dtype=np.float32)
    counts = np.zeros(M, dtype=np.int32)
    np.add.at(out_pts, assignment, points)
    np.add.at(out_col, assignment, colors)
    np.add.at(counts, assignment, 1)
    out_pts /= counts[:, None]
    out_col /= counts[:, None]
    return AggregatedCandidates(positions=out_pts, colors=out_col, voxel_size=voxel_size)
