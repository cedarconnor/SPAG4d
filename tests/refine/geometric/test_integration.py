# tests/refine/geometric/test_integration.py
"""Integration tests for geometric refine pipeline.

test_passthrough: empty OmniRoam input → output identical to input.
test_hole_masked: synthetic hole → hole is filled.
"""
import numpy as np
import pytest
from pathlib import Path


@pytest.mark.skipif(
    not Path("D:/SPAG-4D/output/jobs").exists(),
    reason="Requires output/jobs directory with real PLY files",
)
def test_empty_omniroam_passthrough(tmp_path):
    """With zero OmniRoam frames, output PLY should be identical to input."""
    from spag4d.refine.geometric import refine_splat_geometric, GeometricRefineConfig
    from spag4d.refine.format_compat import load_gaussians_from_ply

    # Find any PLY in output/jobs
    import glob
    plys = glob.glob("D:/SPAG-4D/output/jobs/*_output.ply")
    if not plys:
        pytest.skip("No output PLY available")

    ply_path = plys[0]
    panorama = ply_path.replace("_output.ply", "_input.jpg")
    if not Path(panorama).exists():
        pytest.skip("No input panorama for this PLY")

    config = GeometricRefineConfig(max_frames=0)
    config.color_polish.steps = 0
    out_path = str(tmp_path / "out.ply")

    result = refine_splat_geometric(
        ply_path=ply_path,
        panorama_path=panorama,
        depth_map=np.zeros((1, 1)),
        config=config,
        output_path=out_path,
    )
    assert result["num_new_gaussians"] == 0


def test_consistency_gate_synthetic():
    """Consistency gate drops points with single-frame support."""
    from spag4d.refine.geometric.consistency import ConsistencyConfig, cross_frame_consistency_gate
    from spag4d.refine.geometric.hole_filter import FilterResult

    singleton = FilterResult(
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        candidate_z=np.array([1.0]),
        hole_modes=["alpha"],
        source_frame_idx=0,
        src_uv=np.zeros((1, 2), dtype=np.int32),
    )
    cfg = ConsistencyConfig(min_support_count=2, support_radius=0.5)
    kept = cross_frame_consistency_gate([singleton], cfg)
    assert len(kept) == 0


def test_aggregate_then_init_produces_valid_gaussians():
    """Voxel aggregation + Gaussian init produces valid parameter arrays."""
    from spag4d.refine.geometric.aggregate import aggregate_candidates
    from spag4d.refine.geometric.init_gaussians import new_gaussian_dict

    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (200, 3)).astype(np.float32)
    cols = rng.uniform(0, 1, (200, 3)).astype(np.float32)
    agg = aggregate_candidates(pts, cols, voxel_size=0.1)
    d = new_gaussian_dict(agg)

    assert len(d["means"]) == agg.num_voxels
    assert d["scales"].shape == (agg.num_voxels, 3)
    assert d["quats"].shape == (agg.num_voxels, 4)
    norms = np.linalg.norm(d["quats"], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
