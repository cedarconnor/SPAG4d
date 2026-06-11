"""Top-level ArtiFixer3D (Shape A) refine pipeline.

Drives the WSL/Docker ArtiFixer3D backend end-to-end and returns a fresh,
corrected standard-3DGS PLY tagged with ``PROVENANCE_ARTIFIXER3D``. Peer to the
OmniRoam path (``pipeline_v2.refine_splat_v2``), which is untouched.
"""

import logging
from pathlib import Path

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
