"""Top-level ArtiFixer3D (Shape A) refine pipeline.

Drives the WSL/Docker ArtiFixer3D backend end-to-end and returns a fresh,
corrected standard-3DGS PLY tagged with ``PROVENANCE_ARTIFIXER3D``. Peer to the
OmniRoam path (``pipeline_v2.refine_splat_v2``), which is untouched.
"""

import logging
from pathlib import Path

import numpy as np

from .artifixer3d_adapter import run_artifixer3d_pipeline
from .provenance import PROVENANCE_ARTIFIXER3D, tag_provenance_by_range

logger = logging.getLogger(__name__)


def refine_splat_artifixer3d(cloud_ply, config, output_path, progress_callback=None) -> dict:
    """Shape-A refine: rebuild the cloud through ArtiFixer3D and return a corrected PLY.

    Args:
        cloud_ply: path to the input SPAG standard-3DGS PLY.
        config: ``ArtiFixer3DConfig`` (must have ``enabled=True`` to actually run).
        output_path: destination path for the corrected standard-3DGS PLY.
        progress_callback: optional ``(stage: str) -> None`` hook (reserved).

    Returns:
        dict with ``refined_ply_path``, ``initial_hole_fraction``,
        ``final_hole_fraction``, ``gaussians_count``, ``backend``.

    [container-verify] The ArtiFixer steps run in WSL/Docker; verified by running
    end-to-end on bell_tower (Phase 1 proved the exported PLY loads via
    format_compat). The bridge-back is just ``export_ply`` + the existing loader.
    """
    from .format_compat import load_gaussians_from_ply, save_gaussians_to_ply

    result = run_artifixer3d_pipeline(cloud_ply, config, config.work_dir)

    # Bridge-back: 3DGRUT export_ply already wrote a standard 3DGS PLY whose
    # conventions match SPAG (Phase 1). Load it, tag the whole cloud, re-save to
    # the requested output_path.
    gaussians = load_gaussians_from_ply(result.refined_ply_path, device="cuda")
    n = gaussians.get_xyz.shape[0]
    tag_provenance_by_range(gaussians, 0, n, PROVENANCE_ARTIFIXER3D)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_gaussians_to_ply(gaussians, str(output_path))

    logger.info(
        "ArtiFixer3D refine done: %d Gaussians, holes %.3f -> %.3f, anchors %d / novel %d",
        n, result.initial_hole_fraction, result.final_hole_fraction,
        result.n_anchor, result.n_novel,
    )

    return {
        "refined_ply_path": str(output_path),
        "initial_hole_fraction": result.initial_hole_fraction,
        "final_hole_fraction": result.final_hole_fraction,
        "gaussians_count": n,
        "backend": "artifixer3d",
    }


def _artifixer_metrics(orig_rgbs, refined_rgbs, orig_holes, refined_holes) -> dict:
    """Pure metric core: anchor PSNR (orig vs refined) + coverage before/after.

    Args:
        orig_rgbs / refined_rgbs: lists of (H, W, 3) float renders at the anchor poses.
        orig_holes / refined_holes: lists of (H, W) hole masks (1.0 = hole).

    Returns:
        ``{"anchor_psnr", "coverage_before", "coverage_after"}``.
    """
    from .validation import compute_coverage, compute_psnr

    psnrs = [compute_psnr(a, b) for a, b in zip(orig_rgbs, refined_rgbs)]
    anchor_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    return {
        "anchor_psnr": anchor_psnr,
        "coverage_before": compute_coverage(orig_holes),
        "coverage_after": compute_coverage(refined_holes),
    }


def validate_artifixer_result(original_ply, refined_ply, config) -> dict:
    """Safety guard: render the original cloud and the ArtiFixer3D result at the
    anchor poses and report anchor PSNR + coverage before/after.

    Because Shape A rebuilds every Gaussian (no per-Gaussian correspondence), the
    guard is render-based: anchors are the low-parallax views that should stay
    faithful, so a large anchor-PSNR drop or a coverage regression flags a bad run.

    [container-verify] Needs the host GSFix3D CUDA rasterizer + both PLYs (the
    refined one comes from a GPU container run). The metric core (_artifixer_metrics)
    is unit-tested separately.
    """
    from .artifixer3d_bridge import select_anchor_indices
    from .camera_rig import generate_camera_rig, render_with_hole_mask
    from .format_compat import load_gaussians_from_ply

    g_orig = load_gaussians_from_ply(str(original_ply), device="cuda")
    g_ref = load_gaussians_from_ply(str(refined_ply), device="cuda")

    xyz = g_orig.get_xyz.detach().cpu().numpy()
    radial = np.linalg.norm(xyz, axis=1)
    median_depth = float(np.median(radial[radial > 0]))
    depth_proxy = np.full((16, 16), median_depth, dtype=np.float32)

    cams = generate_camera_rig(
        origin=np.zeros(3, dtype=np.float32), depth_map=depth_proxy,
        num_directions=config.num_directions, num_depths=len(config.translation_fracs),
        fov_deg=config.fov_deg, translation_fracs=tuple(config.translation_fracs),
        resolution=config.render_resolution,
    )

    orig_rgbs, orig_holes, hole_fracs = [], [], []
    for cam in cams:
        rgb, hole = render_with_hole_mask(g_orig, cam, alpha_threshold=0.05)
        orig_rgbs.append(rgb)
        orig_holes.append(hole)
        hole_fracs.append(float(hole.mean()))

    anchors = select_anchor_indices(hole_fracs, config.anchor_parallax_quantile)
    a_orig_rgbs = [orig_rgbs[i] for i in anchors]
    a_orig_holes = [orig_holes[i] for i in anchors]

    a_ref_rgbs, a_ref_holes = [], []
    for i in anchors:
        rgb, hole = render_with_hole_mask(g_ref, cams[i], alpha_threshold=0.05)
        a_ref_rgbs.append(rgb)
        a_ref_holes.append(hole)

    return _artifixer_metrics(a_orig_rgbs, a_ref_rgbs, a_orig_holes, a_ref_holes)
