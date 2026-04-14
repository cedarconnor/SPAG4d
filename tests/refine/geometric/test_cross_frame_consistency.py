# tests/refine/geometric/test_cross_frame_consistency.py
import numpy as np
import pytest
from spag4d.refine.geometric.consistency import (
    ConsistencyConfig,
    cross_frame_consistency_gate,
)
from spag4d.refine.geometric.hole_filter import FilterResult


def _make_result(points, frame_idx, mode="alpha"):
    n = len(points)
    return FilterResult(
        points=np.array(points, dtype=np.float32),
        candidate_z=np.ones(n, dtype=np.float32),
        hole_modes=[mode] * n,
        source_frame_idx=frame_idx,
        src_uv=np.zeros((n, 2), dtype=np.int32),
    )


def test_single_frame_invention_dropped():
    # One frame sees a fake surface; no others agree
    r0 = _make_result([[0.0, 0.0, 0.0]], frame_idx=0)
    r1 = _make_result([[10.0, 10.0, 10.0]], frame_idx=1)  # far away
    r2 = _make_result([[20.0, 20.0, 20.0]], frame_idx=2)

    cfg = ConsistencyConfig(min_support_count=2, support_radius=0.5)
    kept = cross_frame_consistency_gate([r0, r1, r2], cfg)
    assert len(kept) == 0


def test_multi_frame_agreement_kept():
    # 3 frames all see the same surface point
    pt = [1.0, 0.5, 2.0]
    r0 = _make_result([pt, [0.0, 0.0, 0.0]], frame_idx=0)
    r1 = _make_result([[pt[0] + 0.05, pt[1], pt[2]]], frame_idx=1)
    r2 = _make_result([[pt[0], pt[1] + 0.05, pt[2]]], frame_idx=2)

    cfg = ConsistencyConfig(min_support_count=2, support_radius=0.2)
    kept = cross_frame_consistency_gate([r0, r1, r2], cfg)
    # The multi-frame cluster should survive
    assert len(kept) > 0


def test_disoccl_strict_mode_requires_extra_support():
    # disoccl mode needs min_support_count + 1
    pt = [1.0, 0.5, 2.0]
    # 2 frames agree on a disocclusion point — need 3 for strict mode
    r0 = _make_result([pt], frame_idx=0, mode="disoccl")
    r1 = _make_result([[pt[0] + 0.05, pt[1], pt[2]]], frame_idx=1, mode="disoccl")

    cfg = ConsistencyConfig(min_support_count=2, support_radius=0.2, strict_mode_for_disoccl=True)
    kept = cross_frame_consistency_gate([r0, r1], cfg)
    assert len(kept) == 0  # only 2 frames agree, needs 3

    # Add a third frame — now it should pass
    r2 = _make_result([[pt[0], pt[1] + 0.05, pt[2]]], frame_idx=2, mode="disoccl")
    kept = cross_frame_consistency_gate([r0, r1, r2], cfg)
    assert len(kept) > 0
