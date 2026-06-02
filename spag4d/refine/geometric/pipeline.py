# spag4d/refine/geometric/pipeline.py
"""refine_splat_geometric — orchestration of all geometric refine stages."""
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .config import GeometricRefineConfig
from .depth_convention import radial_to_z, assert_is_z_depth
from .erp_unproject import unproject_erp_depth_to_points
from .masks import sky_mask_hsv, specular_mask, build_validity_mask
from .depth_align import align_depth_irls, AlignmentResult
from .render_utils import render_base_from_pose
from .hole_filter import HoleFilterConfig, filter_candidate_points_per_frame
from .consistency import cross_frame_consistency_gate
from .aggregate import aggregate_candidates
from .init_gaussians import new_gaussian_dict, estimate_base_voxel_size
from .diagnostics import FrameDiagnostics, PipelineDiagnostics, write_diagnostics_json

logger = logging.getLogger("spag4d.refine.geometric")


def refine_splat_geometric(
    ply_path: str,
    panorama_path: str,
    depth_map: np.ndarray,
    config: GeometricRefineConfig = None,
    output_path: str = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    diagnostics_dir: Optional[str] = None,
) -> dict:
    """Geometric OmniRoam refine pipeline.

    Stages:
      0. OmniRoam generation (reuse existing pipeline_v2 Stage 0-3)
      1. Per-frame depth estimation via DA360
      2. Per-frame depth alignment to base splat render
      3. Per-frame hole filtering (alpha + disocclusion gates)
      3.5. Cross-frame consistency gate
      4. Voxel aggregation
      5. Gaussian initialization + injection
      6. Color polish (optional)

    Returns dict with output_ply_path, num_new_gaussians, total_runtime_seconds.
    """
    if config is None:
        config = GeometricRefineConfig()

    t0 = time.time()

    # --- Stage 0: Load base splat + run OmniRoam (reuse pipeline_v2 stages) ---
    from spag4d.refine.format_compat import load_gaussians_from_ply
    from spag4d.refine.pipeline_v2 import _run_omniroam_stages

    logger.info("[geometric] Loading base splat from %s", ply_path)
    base_gaussians = load_gaussians_from_ply(ply_path)

    logger.info("[geometric] Running OmniRoam stages")
    omniroam_result = _run_omniroam_stages(
        panorama_path=panorama_path,
        ply_path=ply_path,
        trajectory_mode=config.trajectory_mode,
        upscale_backend=config.upscale_backend,
        max_frames=config.max_frames,
        progress_callback=progress_callback,
    )
    frames = omniroam_result["frames"]           # list of (H, W, 3) uint8 np arrays
    poses = omniroam_result["poses"]             # (N, 4, 4)

    if not frames:
        logger.warning("[geometric] No OmniRoam frames generated; returning base splat unchanged")
        return {"output_ply_path": ply_path, "num_new_gaussians": 0, "total_runtime_seconds": time.time() - t0}

    # --- Stage 1: Per-frame depth estimation ---
    logger.info("[geometric] Stage 1: depth estimation for %d frames", len(frames))
    from spag4d.da360_model import DA360Model
    depth_model = DA360Model()
    depth_model.load()

    frame_depths = []
    frame_sky_masks = []
    frame_spec_masks = []
    for i, frame in enumerate(frames):
        frame_u8 = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
        d_raw = depth_model.estimate_depth(frame_u8.astype(np.float32) / 255.0)
        frame_depths.append(d_raw)
        frame_sky_masks.append(sky_mask_hsv(frame_u8))
        frame_spec_masks.append(specular_mask(frame_u8))

    # --- Estimate voxel size from base splat ---
    base_xyz = base_gaussians.get_xyz.detach().cpu().numpy()
    voxel_size = config.voxel_size or estimate_base_voxel_size(base_xyz)
    logger.info("[geometric] voxel_size=%.4f", voxel_size)

    # --- Stages 2, 3, 3.5: per-frame alignment + filtering ---
    H_erp, W_erp = frames[0].shape[:2]
    hole_filter_cfg = HoleFilterConfig(
        alpha_hole_threshold=config.alpha_hole_threshold,
        alpha_confident_threshold=config.alpha_confident_threshold,
        depth_disoccl_margin_ratio=config.depth_disoccl_margin_ratio,
    )

    per_frame_results = []
    frame_diags = []

    for i, (frame, pose, d_raw, sky, spec) in enumerate(
        zip(frames, poses, frame_depths, frame_sky_masks, frame_spec_masks)
    ):
        t_frame = time.time()
        logger.info("[geometric] Frame %d/%d", i + 1, len(frames))

        # Render base splat from this pose
        render = render_base_from_pose(base_gaussians, pose, (H_erp, W_erp))

        # Build validity mask
        vmask = build_validity_mask(
            render.alpha, render.depth, sky, spec,
            alpha_confident=config.alpha_confident_threshold,
            gradient_threshold=config.depth_gradient_threshold,
        )

        # Flatten for alignment
        valid_d = d_raw[vmask].ravel()
        valid_r = render.depth[vmask].ravel()

        align = align_depth_irls(
            valid_d, valid_r, np.ones(len(valid_d), dtype=bool),
            mode=config.align_mode,
            min_inlier_fraction=config.min_inlier_fraction,
        )

        if not align.converged:
            logger.warning("[geometric] Frame %d alignment failed — skipping", i)
            frame_diags.append(FrameDiagnostics(
                frame_idx=i, alignment=align,
                num_candidates=0, num_kept_alpha=0, num_kept_disoccl=0,
                num_candidates_pre_consistency=0, num_candidates_post_consistency=0,
                consistency_drop_rate=0.0, avg_support_count=0.0,
                skipped=True, skip_reason="alignment_failed",
                wall_time_seconds=time.time() - t_frame,
            ))
            continue

        # Apply alignment and convert radial → z-depth
        d_aligned_radial = align.scale * d_raw + align.shift

        # radial → z using lat/lon grid
        v_idx, u_idx = np.mgrid[0:H_erp, 0:W_erp]
        lon = (u_idx / W_erp - 0.5) * 2.0 * np.pi
        lat = (0.5 - v_idx / H_erp) * np.pi
        d_z = radial_to_z(d_aligned_radial, lat, lon)

        assert_is_z_depth(d_z, render.depth)

        # Unproject to world-space candidates (all pixels with positive depth)
        cand_pts = unproject_erp_depth_to_points(d_z, pose)
        n_cand = len(cand_pts)

        if n_cand == 0:
            continue

        # src pixel coords for filter
        v_flat, u_flat = np.where(d_z > 0.01)
        src_uv = np.stack([u_flat, v_flat], axis=-1).astype(np.int32)
        cand_z = d_z[v_flat, u_flat]

        filter_result = filter_candidate_points_per_frame(
            candidates=cand_pts,
            candidate_z=cand_z,
            source_frame_idx=i,
            rendered_depth=render.depth,
            rendered_alpha=render.alpha,
            src_uv=src_uv,
            config=hole_filter_cfg,
        )

        per_frame_results.append(filter_result)

        n_alpha = sum(1 for m in filter_result.hole_modes if m == "alpha")
        n_disoccl = filter_result.num_kept - n_alpha

        frame_diags.append(FrameDiagnostics(
            frame_idx=i, alignment=align,
            num_candidates=n_cand,
            num_kept_alpha=n_alpha, num_kept_disoccl=n_disoccl,
            num_candidates_pre_consistency=filter_result.num_kept,
            num_candidates_post_consistency=0,  # filled after consistency gate
            consistency_drop_rate=0.0, avg_support_count=0.0,
            skipped=False, skip_reason=None,
            wall_time_seconds=time.time() - t_frame,
        ))

    # --- Stage 3.5: Cross-frame consistency ---
    logger.info("[geometric] Stage 3.5: cross-frame consistency gate")
    survivors = cross_frame_consistency_gate(
        per_frame_results, config.consistency, voxel_size=voxel_size
    )

    n_pre = sum(r.num_kept for r in per_frame_results)
    n_post = len(survivors)
    drop_rate = 1.0 - (n_post / max(n_pre, 1))
    traj_warn = drop_rate > 0.30
    if traj_warn:
        logger.warning("[geometric] High consistency drop rate %.1f%% — consider trajectory_mode='all'", drop_rate * 100)

    # --- Stage 4: Aggregation ---
    if n_post == 0:
        logger.warning("[geometric] No candidates survived consistency gate")
        survivors_colors = np.zeros((0, 3), dtype=np.float32)
    else:
        # Collect colors for survivors (use frame colors from OmniRoam frames)
        # For now use white as placeholder; refined in color polish
        survivors_colors = np.ones((n_post, 3), dtype=np.float32) * 0.5

    logger.info("[geometric] Stage 4: aggregating %d survivors", n_post)
    if n_post > 0:
        aggregated = aggregate_candidates(
            survivors, survivors_colors, voxel_size=voxel_size,
            max_gaussians=config.max_new_gaussians
        )
    else:
        from .aggregate import AggregatedCandidates
        aggregated = AggregatedCandidates(
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
            voxel_size=voxel_size,
        )

    # --- Stage 5: Gaussian initialization ---
    logger.info("[geometric] Stage 5: initializing %d new Gaussians", aggregated.num_voxels)
    new_g = new_gaussian_dict(aggregated)

    # --- Merge with base splat and write PLY ---
    out_path = output_path or str(Path(ply_path).with_suffix("")) + "_geometric_refined.ply"
    _merge_and_save(base_gaussians, new_g, out_path)

    # --- Stage 6: Color polish (if steps > 0) ---
    if config.color_polish.steps > 0 and aggregated.num_voxels > 0:
        logger.info("[geometric] Stage 6: color polish (%d steps)", config.color_polish.steps)
        from .color_polish import color_polish
        color_polish(out_path, panorama_path, config.color_polish, len(base_xyz))

    total_time = time.time() - t0

    # Write diagnostics
    if config.write_diagnostics:
        diag_dir = Path(diagnostics_dir) if diagnostics_dir else Path(out_path).parent
        diag_dir.mkdir(parents=True, exist_ok=True)
        pipeline_diag = PipelineDiagnostics(
            total_frames=len(frames),
            frames_skipped=sum(1 for d in frame_diags if d.skipped),
            total_new_gaussians=aggregated.num_voxels,
            trajectory_coverage_warning=traj_warn,
            scale_consistency_std_ratio=0.0,  # TODO: compute from align results
            color_polish_seam_delta=None,
            total_runtime_seconds=total_time,
        )
        write_diagnostics_json(frame_diags, pipeline_diag,
                               diag_dir / "geometric_diagnostics.json")

    logger.info("[geometric] Done. %d new Gaussians in %.1fs", aggregated.num_voxels, total_time)
    return {
        "output_ply_path": out_path,
        "num_new_gaussians": aggregated.num_voxels,
        "total_runtime_seconds": total_time,
    }


def _merge_and_save(base_gaussians, new_g: dict, out_path: str) -> None:
    """Concatenate base splat + new Gaussians and write PLY."""
    import torch
    from spag4d.ply_writer import save_ply_gsplat

    base_xyz = base_gaussians.get_xyz.detach().cpu().numpy()
    base_dc = base_gaussians.get_features[:, 0, :].detach().cpu().numpy()  # (N, 3)
    base_scaling = torch.exp(base_gaussians.get_scaling).detach().cpu().numpy()
    base_rot = base_gaussians.get_rotation.detach().cpu().numpy()
    base_opacity = torch.sigmoid(base_gaussians.get_opacity).detach().cpu().numpy().squeeze(-1)

    if len(new_g["means"]) > 0:
        means = np.vstack([base_xyz, new_g["means"]])
        # New colors in SH0 form: (c - 0.5) / (1/(4pi))^0.5
        sh0_scale = (1.0 / (4.0 * np.pi)) ** 0.5
        new_sh0 = (new_g["colors"] - 0.5) / sh0_scale
        colors = np.vstack([base_dc, new_sh0])
        scales = np.vstack([base_scaling, new_g["scales"]])
        quats = np.vstack([base_rot, new_g["quats"]])
        # New opacity from logit storage
        new_opac = 1.0 / (1.0 + np.exp(-new_g["opacities"]))
        opacities = np.concatenate([base_opacity, new_opac])
    else:
        means, colors, scales, quats = base_xyz, base_dc, base_scaling, base_rot
        opacities = base_opacity

    gaussians_dict = {
        "means": means,
        "colors": colors,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
    }
    save_ply_gsplat(gaussians_dict, out_path, sh_degree=0, colors_linear=False)
    logger.info("[geometric] Saved %d Gaussians to %s", len(means), out_path)
