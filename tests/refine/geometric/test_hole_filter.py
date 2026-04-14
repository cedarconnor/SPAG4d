# tests/refine/geometric/test_hole_filter.py
import numpy as np
from spag4d.refine.geometric.hole_filter import (
    HoleFilterConfig,
    FilterResult,
    filter_candidate_points_per_frame,
)


def _make_scene(H=64, W=128):
    alpha = np.ones((H, W), dtype=np.float32) * 0.95  # mostly confident
    depth = np.full((H, W), 3.0, dtype=np.float32)
    return alpha, depth


def test_low_alpha_region_kept_as_alpha_mode():
    H, W = 64, 128
    alpha, depth = _make_scene(H, W)
    alpha[:, :10] = 0.1  # clear hole on left strip

    # Candidates in the hole strip
    cand_pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
    cand_z = np.array([1.0, 1.0], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    # Map these to pixels in the hole region
    src_uv = np.array([[2, 32], [5, 32]], dtype=np.int32)  # (u, v) in hole

    cfg = HoleFilterConfig()
    result = filter_candidate_points_per_frame(
        candidates=cand_pts,
        candidate_z=cand_z,
        source_frame_idx=0,
        rendered_depth=depth,
        rendered_alpha=alpha,
        src_uv=src_uv,
        config=cfg,
    )
    assert result.num_kept > 0
    assert all(m == "alpha" for m in result.hole_modes)


def test_confident_region_behind_surface_rejected():
    H, W = 64, 128
    alpha, depth = _make_scene(H, W)
    # Candidate behind rendered surface, confident region
    cand_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    cand_z = np.array([5.0], dtype=np.float32)  # behind rendered depth=3
    pose = np.eye(4, dtype=np.float32)
    src_uv = np.array([[64, 32]], dtype=np.int32)  # confident region

    cfg = HoleFilterConfig()
    result = filter_candidate_points_per_frame(
        candidates=cand_pts,
        candidate_z=cand_z,
        source_frame_idx=0,
        rendered_depth=depth,
        rendered_alpha=alpha,
        src_uv=src_uv,
        config=cfg,
    )
    assert result.num_kept == 0


def test_disocclusion_nearer_point_kept():
    H, W = 64, 128
    alpha = np.full((H, W), 0.5, dtype=np.float32)  # medium alpha band
    depth = np.full((H, W), 3.0, dtype=np.float32)

    # Candidate at z=1.0, nearer than rendered z=3.0 → disocclusion
    cand_pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    cand_z = np.array([1.0], dtype=np.float32)
    src_uv = np.array([[64, 32]], dtype=np.int32)

    cfg = HoleFilterConfig()
    result = filter_candidate_points_per_frame(
        candidates=cand_pts,
        candidate_z=cand_z,
        source_frame_idx=0,
        rendered_depth=depth,
        rendered_alpha=alpha,
        src_uv=src_uv,
        config=cfg,
    )
    assert result.num_kept == 1
    assert result.hole_modes[0] == "disoccl"
