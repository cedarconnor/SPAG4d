"""Refine v2 pipeline orchestrator — 7-stage OmniRoam refinement flow.

Wires all refine-v2 modules together:

    Stage 0: Load initial splat (input PLY) + panorama
    Stage 1: Gap analysis — render from eval cameras, classify directions
    Stage 2: OmniRoam generation (WSL2 subprocess per selected trajectory)
    Stage 3: Upscale (OPTIONAL — skipped in Phase 1, upscale_backend="none")
    Stage 4: Gap-directed view selection — perspective crops, filter by gap ratio
    Stage 4.5: Gap seeding — seed sparse Gaussians from source-pano depth
    Stage 5: Confidence-masked splat optimization — distill with tier-1 + tier-2
    Stage 6: Validation & export — source-anchor PSNR check, coverage measurement
"""

import glob
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .omniroam_config import OmniRoamConfig
from .omniroam_adapter import validate_wsl_environment, run_omniroam_wsl, extract_video_frames
from .omniroam_trajectory import generate_omniroam_trajectory
from .gap_analysis import classify_gap_directions, select_trajectories
from .view_selector import extract_perspective_crop, compute_perspective_pose, filter_views_by_gap
from .scale_alignment import parse_scale_config, estimate_scale_factor
from .gap_seeding import seed_gap_gaussians
from ..seedvr2 import upscale_video as seedvr2_upscale_video, SeedVR2Config, validate_seedvr2_environment
from .validation import compute_psnr, compute_coverage, check_source_anchor
from .camera_rig import (
    generate_camera_rig, render_with_hole_mask, extract_cubemap_views, CameraPose,
)
from .distill import distill_to_gaussians, compute_weighted_loss
from .provenance import (
    tag_gaussian_provenance, tag_provenance_by_range,
    PROVENANCE_ORIGINAL, PROVENANCE_GAP_SEED, PROVENANCE_OMNIROAM,
)
from .format_compat import load_gaussians_from_ply, save_gaussians_to_ply

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_result(output_path, initial_hole, final_hole, count, iters, start_time, psnr):
    """Assemble the standard result dict returned by refine_splat_v2."""
    total_time = time.time() - start_time
    return {
        "refined_ply_path": output_path,
        "initial_hole_fraction": initial_hole,
        "final_hole_fraction": final_hole,
        "gaussians_count": count,
        "iterations_used": iters,
        "total_time": round(total_time, 1),
        "source_anchor_psnr": psnr,
    }


def _inject_seed_gaussians(gaussians, seed_result):
    """Add seeded Gaussians to an existing GaussianModel.

    Extends all parameter tensors so the model can be optimised with the
    new seeds in-place.
    """
    import torch

    positions = torch.from_numpy(seed_result["positions"]).cuda()
    n = positions.shape[0]
    if n == 0:
        return

    device = gaussians.get_xyz.device

    # Neutral SH (gray)
    sh_dim = gaussians._features_dc.shape[1:]
    new_features_dc = torch.zeros(n, *sh_dim, device=device)

    # Small isotropic scale (log space)
    new_scaling = torch.full((n, 3), -5.0, device=device)

    # Identity rotation quaternion
    new_rotation = torch.zeros(n, 4, device=device)
    new_rotation[:, 0] = 1.0

    # Low opacity (logit space)
    opacity_val = seed_result["opacities"][0] if len(seed_result["opacities"]) > 0 else 0.01
    logit_opacity = float(np.log(opacity_val / (1.0 - opacity_val)))
    new_opacity = torch.full((n, 1), logit_opacity, device=device)

    # Extend model tensors
    gaussians._xyz = torch.nn.Parameter(
        torch.cat([gaussians._xyz, positions], dim=0))
    gaussians._features_dc = torch.nn.Parameter(
        torch.cat([gaussians._features_dc, new_features_dc], dim=0))
    if hasattr(gaussians, '_features_rest') and gaussians._features_rest is not None:
        rest_dim = gaussians._features_rest.shape[1:]
        new_rest = torch.zeros(n, *rest_dim, device=device)
        gaussians._features_rest = torch.nn.Parameter(
            torch.cat([gaussians._features_rest, new_rest], dim=0))
    gaussians._scaling = torch.nn.Parameter(
        torch.cat([gaussians._scaling, new_scaling], dim=0))
    gaussians._rotation = torch.nn.Parameter(
        torch.cat([gaussians._rotation, new_rotation], dim=0))
    gaussians._opacity = torch.nn.Parameter(
        torch.cat([gaussians._opacity, new_opacity], dim=0))

    # Extend auxiliary tensors
    gaussians.max_radii2D = torch.cat([
        gaussians.max_radii2D, torch.zeros(n, device=device)])
    gaussians.xyz_gradient_accum = torch.cat([
        gaussians.xyz_gradient_accum, torch.zeros(n, 1, device=device)])
    gaussians.denom = torch.cat([
        gaussians.denom, torch.zeros(n, 1, device=device)])

    logger.info(f"Injected {n} seed Gaussians into model "
                f"(total now {gaussians._xyz.shape[0]})")


def _render_eval_cameras(gaussians, cameras, alpha_threshold=0.1):
    """Render from evaluation cameras and return hole masks + azimuths.

    Returns:
        hole_masks: list of (H, W) float32 arrays
        azimuths_deg: list of floats
    """
    hole_masks = []
    azimuths_deg = []
    for cam in cameras:
        _rgb, mask = render_with_hole_mask(gaussians, cam, alpha_threshold=alpha_threshold)
        hole_masks.append(mask)
        # Compute azimuth from camera position (position relative to origin)
        dx, dz = cam.position[0], cam.position[2]
        azimuth = float(np.degrees(np.arctan2(dz, dx))) % 360.0
        azimuths_deg.append(azimuth)
    return hole_masks, azimuths_deg


def _build_aggregate_gap_mask(hole_masks, azimuths_deg, erp_h, erp_w):
    """Build an ERP-shaped gap mask by back-projecting hole masks into panorama space.

    This is a simplified aggregate: for each eval camera direction we mark
    the corresponding horizontal band of the ERP as a gap where the average
    hole fraction exceeds a threshold.

    Returns:
        gap_mask: (erp_h, erp_w) boolean array
    """
    gap_mask = np.zeros((erp_h, erp_w), dtype=bool)
    for mask, az_deg in zip(hole_masks, azimuths_deg):
        frac = float(np.mean(mask))
        if frac < 0.03:
            continue
        # Map azimuth to ERP column range (approximate 90-deg sector)
        # SPAG convention: pixel_x = (1 - theta/(2*pi)) * (w-1)
        # theta in radians from azimuth
        az_rad = np.radians(az_deg)
        center_theta = az_rad % (2 * np.pi)
        half_span = np.pi / 4  # 45 degrees either side
        theta_lo = (center_theta - half_span) % (2 * np.pi)
        theta_hi = (center_theta + half_span) % (2 * np.pi)

        col_lo = int((1.0 - theta_hi / (2 * np.pi)) * (erp_w - 1))
        col_hi = int((1.0 - theta_lo / (2 * np.pi)) * (erp_w - 1))

        if col_lo <= col_hi:
            gap_mask[:, col_lo:col_hi + 1] = True
        else:
            # Wraps around
            gap_mask[:, col_lo:] = True
            gap_mask[:, :col_hi + 1] = True

    return gap_mask


def _save_diagnostics_v2(diag_dir, stage, data):
    """Save diagnostic images for a given pipeline stage."""
    from PIL import Image

    diag_dir = Path(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    if stage == "gap_report" and "report" in data:
        report = data["report"]
        with open(diag_dir / "gap_report.txt", "w") as f:
            f.write(f"avg_hole_fraction: {report.avg_hole_fraction:.4f}\n")
            f.write(f"worst_direction: {report.worst_direction}\n")
            f.write(f"converged: {report.converged}\n")
            for d, v in report.per_direction_fractions.items():
                f.write(f"  {d}: {v:.4f}\n")
            f.write(f"recommended_trajectories: {report.recommended_trajectories}\n")

    if stage == "views" and "views" in data:
        for i, v in enumerate(data["views"][:16]):
            if "crop" in v:
                img = Image.fromarray(
                    (v["crop"] * 255).clip(0, 255).astype(np.uint8))
                img.save(diag_dir / f"view_{i:03d}_yaw{v.get('yaw_deg', 0):.0f}.png")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def refine_splat_v2(
    ply_path: str,
    panorama_path: str,
    depth_map: np.ndarray,
    config: OmniRoamConfig = None,
    output_path: str = None,
    progress_callback: Optional[Callable] = None,
    diagnostics_dir: Optional[str] = None,
) -> dict:
    """Full OmniRoam-based refinement pipeline (v2).

    Seven-stage flow that analyses gaps in the initial splat, generates
    novel views via OmniRoam, and distils them back into the Gaussian model.

    Args:
        ply_path: Path to the initial SPAG-4D PLY file.
        panorama_path: Path to the source equirectangular panorama image.
        depth_map: (H, W) float32 radial depth map from DAP/DA360.
        config: OmniRoamConfig controlling all pipeline knobs.
        output_path: Where to save the refined PLY. Defaults to ``<input>_v2.ply``.
        progress_callback: Optional ``(stage: str, pct: int) -> None``.
        diagnostics_dir: Optional directory to save intermediate images.

    Returns:
        Dict with keys: refined_ply_path, initial_hole_fraction,
        final_hole_fraction, gaussians_count, iterations_used,
        total_time, source_anchor_psnr.
    """
    config = config or OmniRoamConfig()
    start_time = time.time()
    output_path = output_path or ply_path.replace(".ply", "_v2.ply")
    diag_dir = Path(diagnostics_dir) if diagnostics_dir else None

    def report(stage, pct):
        logger.info(f"[refine-v2] stage={stage} pct={pct}")
        if progress_callback:
            progress_callback(stage, pct)

    # ── Stage 0: Load initial splat + panorama ──────────────────────────
    report("load", 0)

    # Copy panorama to output directory so it survives temp file cleanup
    # during long-running OmniRoam generation
    import shutil
    safe_pano_dir = Path(output_path).parent / f"_refine_v2_{Path(output_path).stem}"
    safe_pano_dir.mkdir(parents=True, exist_ok=True)
    safe_pano_path = safe_pano_dir / Path(panorama_path).name
    shutil.copy2(panorama_path, safe_pano_path)
    panorama_path = str(safe_pano_path)
    logger.info(f"[Stage 0] Panorama preserved at {panorama_path}")

    from PIL import Image
    panorama = np.array(Image.open(panorama_path)).astype(np.float32) / 255.0
    if panorama.ndim == 3 and panorama.shape[2] == 4:
        panorama = panorama[:, :, :3]
    erp_h, erp_w = panorama.shape[:2]

    gaussians = load_gaussians_from_ply(ply_path, device="cuda")
    initial_gaussian_count = gaussians.get_xyz.shape[0]

    # Tag all loaded Gaussians as original
    tag_provenance_by_range(gaussians, 0, initial_gaussian_count, PROVENANCE_ORIGINAL)

    logger.info(f"[Stage 0] Loaded {initial_gaussian_count} Gaussians, "
                f"panorama {erp_w}x{erp_h}")

    # ── Stage 1: Gap analysis ───────────────────────────────────────────
    report("gap_analysis", 5)

    eval_cameras = generate_camera_rig(
        origin=np.array([0.0, 0.0, 0.0]),
        depth_map=depth_map,
        num_directions=12,
        num_depths=3,
    )
    hole_masks, azimuths_deg = _render_eval_cameras(
        gaussians, eval_cameras, alpha_threshold=config.hole_mask_threshold,
    )

    gap_report = classify_gap_directions(
        hole_masks, azimuths_deg,
        min_hole_fraction=config.convergence_threshold,
    )

    initial_hole_fraction = gap_report.avg_hole_fraction
    logger.info(f"[Stage 1] Gap analysis: avg_hole={initial_hole_fraction:.4f}, "
                f"worst={gap_report.worst_direction}, "
                f"converged={gap_report.converged}")

    if diag_dir:
        _save_diagnostics_v2(diag_dir, "gap_report", {"report": gap_report})

    # Early exit if already converged
    if gap_report.converged:
        logger.info("[Stage 1] Splat already converged — no refinement needed")
        save_gaussians_to_ply(gaussians, output_path)
        return _build_result(
            output_path, initial_hole_fraction, initial_hole_fraction,
            initial_gaussian_count, 0, start_time, psnr=100.0,
        )

    # ── Stage 2: OmniRoam generation ────────────────────────────────────
    report("omniroam", 15)

    trajectories = select_trajectories(gap_report, config)
    logger.info(f"[Stage 2] Selected trajectories: {trajectories}")

    omniroam_frames_by_traj = {}  # preset -> list of (H,W,3) frames
    omniroam_translations_by_traj = {}  # preset -> list of (3,) translations

    if trajectories and config.enabled:
        validate_wsl_environment(config)

        # Unique work dir per run to avoid stale files from previous runs
        import hashlib
        run_hash = hashlib.md5(output_path.encode()).hexdigest()[:8]
        work_dir = Path(output_path).parent / f"_omniroam_work_{run_hash}"
        work_dir.mkdir(parents=True, exist_ok=True)

        for preset in trajectories:
            report(f"omniroam_{preset}", 15 + trajectories.index(preset) * 5)
            logger.info(f"[Stage 2] Running OmniRoam preset={preset}")

            traj_dir = work_dir / preset
            traj_dir.mkdir(parents=True, exist_ok=True)

            run_omniroam_wsl(
                image_path=panorama_path,
                output_dir=str(traj_dir),
                preset=preset,
                config=config,
                progress_callback=None,
            )

            # Find the output video
            videos = glob.glob(str(traj_dir / "**" / "generated.mp4"), recursive=True)
            if not videos:
                logger.warning(f"[Stage 2] No video output for preset={preset}")
                continue

            frames = extract_video_frames(videos[0])
            logger.info(f"[Stage 2] Extracted {len(frames)} frames for preset={preset}")

            # Generate trajectory translations
            _cam_traj, translations = generate_omniroam_trajectory(
                preset=preset,
                step_m=config.step_m,
                amp_m=config.s_curve_amp_m,
                loop_radius_m=config.loop_radius_m,
                num_video_frames=len(frames),
            )

            omniroam_frames_by_traj[preset] = frames
            omniroam_translations_by_traj[preset] = translations

    elif trajectories and not config.enabled:
        logger.info("[Stage 2] OmniRoam disabled in config — skipping generation")

    # ── Stage 3: Upscale (OPTIONAL) ────────────────────────────────────
    report("upscale", 35)

    logger.info(f"[Stage 3] upscale_backend={config.upscale_backend}, "
                f"frames_available={bool(omniroam_frames_by_traj)}")

    if config.upscale_backend == "seedvr2" and omniroam_frames_by_traj:
        logger.info(f"[Stage 3] Upscaling with SeedVR2 "
                     f"(resolution={config.seedvr2_target_resolution})")
        try:
            seedvr2_cfg = SeedVR2Config(
                model=config.seedvr2_model,
                target_resolution=config.seedvr2_target_resolution,
                color_correction=config.seedvr2_color_correction,
                block_swap=config.seedvr2_block_swap,
            )
            validate_seedvr2_environment(seedvr2_cfg)

            for preset in list(omniroam_frames_by_traj.keys()):
                report(f"upscale_{preset}", 35)

                traj_dir = work_dir / preset
                videos = list(traj_dir.rglob("generated.mp4"))
                if not videos:
                    logger.warning(f"[Stage 3] No video found for preset={preset}, skipping upscale")
                    continue

                src_video = str(videos[0])
                upscaled_video = str(videos[0].parent / "generated_upscaled.mp4")
                logger.info(f"[Stage 3] Upscaling {src_video}")

                seedvr2_upscale_video(
                    video_path=src_video,
                    output_path=upscaled_video,
                    config=seedvr2_cfg,
                )

                upscaled_frames = extract_video_frames(upscaled_video)
                if upscaled_frames:
                    logger.info(f"[Stage 3] Upscaled {preset}: {len(upscaled_frames)} frames "
                                f"at {upscaled_frames[0].shape[1]}x{upscaled_frames[0].shape[0]}")
                    omniroam_frames_by_traj[preset] = upscaled_frames
                else:
                    logger.warning(f"[Stage 3] Failed to extract upscaled frames for {preset}")
        except Exception as e:
            logger.error(f"[Stage 3] SeedVR2 upscale FAILED: {e}", exc_info=True)
            raise

    elif config.upscale_backend != "none":
        logger.warning(f"[Stage 3] Unknown upscale backend: {config.upscale_backend}")
    else:
        logger.info("[Stage 3] Upscale skipped (backend=none)")

    # ── Stage 4: Gap-directed view selection ────────────────────────────
    report("view_selection", 40)

    candidate_views = []
    tier2_images = []
    tier2_cameras = []

    for preset, frames in omniroam_frames_by_traj.items():
        translations = omniroam_translations_by_traj[preset]
        directions = config.extract_directions  # e.g. [0, 90, 180, 270]

        for frame_idx, frame in enumerate(frames):
            if frame_idx >= len(translations):
                break
            translation = translations[frame_idx]

            for yaw_deg in directions:
                # Compute the perspective camera pose for this view
                cam_pose = compute_perspective_pose(
                    translation=translation,
                    yaw_deg=yaw_deg,
                    fov_deg=config.extract_fov_degrees,
                    size=256,
                )

                # Render the current splat from this viewpoint to get gap ratio
                _rgb, hole_mask = render_with_hole_mask(
                    gaussians, cam_pose, alpha_threshold=config.hole_mask_threshold,
                )
                gap_ratio = float(np.mean(hole_mask))

                # Extract the perspective crop from the OmniRoam frame
                crop = extract_perspective_crop(
                    frame, yaw_deg=yaw_deg,
                    fov_deg=config.extract_fov_degrees,
                    size=256,
                )

                candidate_views.append({
                    "crop": crop,
                    "camera": cam_pose,
                    "gap_ratio": gap_ratio,
                    "hole_mask": hole_mask,
                    "preset": preset,
                    "frame_idx": frame_idx,
                    "yaw_deg": yaw_deg,
                })

    logger.info(f"[Stage 4] Built {len(candidate_views)} candidate views")

    # Filter by gap ratio
    selected_views = filter_views_by_gap(
        candidate_views,
        min_gap_ratio=config.min_gap_ratio,
        max_views=config.max_omniroam_views,
    )
    logger.info(f"[Stage 4] Selected {len(selected_views)} views after gap filtering")

    if diag_dir:
        _save_diagnostics_v2(diag_dir, "views", {"views": selected_views})

    # Collect tier-2 supervision data from selected views
    for view in selected_views:
        tier2_images.append(view["crop"])
        tier2_cameras.append(view["camera"])

    # ── Stage 4.5: Gap seeding ──────────────────────────────────────────
    report("gap_seeding", 50)

    # Build aggregate gap mask from eval camera renders
    aggregate_gap_mask = _build_aggregate_gap_mask(
        hole_masks, azimuths_deg, erp_h, erp_w,
    )
    gap_pixel_count = int(aggregate_gap_mask.sum())
    logger.info(f"[Stage 4.5] Aggregate gap mask: {gap_pixel_count} gap pixels "
                f"({gap_pixel_count / (erp_h * erp_w) * 100:.1f}%)")

    pre_seed_count = gaussians.get_xyz.shape[0]

    if gap_pixel_count > 0:
        seed_result = seed_gap_gaussians(
            source_depth=depth_map,
            gap_mask=aggregate_gap_mask,
            stride=config.gap_seed_stride,
            initial_opacity=config.gap_seed_initial_opacity,
        )

        n_seeds = seed_result["positions"].shape[0]
        if n_seeds > 0:
            _inject_seed_gaussians(gaussians, seed_result)
            # Tag the newly injected seeds
            tag_provenance_by_range(
                gaussians, pre_seed_count, pre_seed_count + n_seeds,
                PROVENANCE_GAP_SEED,
            )
            logger.info(f"[Stage 4.5] Injected {n_seeds} gap-seed Gaussians")
    else:
        logger.info("[Stage 4.5] No gap pixels — skipping seeding")

    # ── Stage 5: Confidence-masked splat optimisation ───────────────────
    report("distill", 55)

    # Tier-1: cubemap views from the source panorama (high confidence)
    cubemap_faces, cubemap_cameras = extract_cubemap_views(
        panorama, depth_map, face_size=512,
    )

    # Scale alignment for OmniRoam poses (Stage 7 in config)
    scale_factor = 1.0
    scale_cfg = parse_scale_config(config.scale_alignment)

    if scale_cfg == "reprojection" and len(tier2_images) > 0:
        logger.info("[Stage 5] Running reprojection-based scale alignment")
        # Use the first OmniRoam view for scale estimation
        first_view = selected_views[0]
        first_cam = first_view["camera"]
        omniroam_ref = first_view["crop"]

        def _render_at_scale(s):
            """Render splat from scaled camera position."""
            scaled_pos = first_cam.position * s
            scaled_cam = CameraPose(
                position=scaled_pos,
                look_at=first_cam.look_at,
                up=first_cam.up.copy(),
                fov_deg=first_cam.fov_deg,
                width=first_cam.width,
                height=first_cam.height,
            )
            rgb, _mask = render_with_hole_mask(gaussians, scaled_cam)
            return rgb

        scale_factor = estimate_scale_factor(
            render_fn=_render_at_scale,
            omniroam_frame1=omniroam_ref,
            search_range=config.scale_search_range,
            num_samples=config.scale_search_samples,
        )
        logger.info(f"[Stage 5] Scale alignment: factor={scale_factor:.4f}")

        # Apply scale to all tier-2 camera positions
        if abs(scale_factor - 1.0) > 1e-4:
            for cam in tier2_cameras:
                cam.position = cam.position * scale_factor

    elif isinstance(scale_cfg, float):
        scale_factor = scale_cfg
        logger.info(f"[Stage 5] Manual scale factor: {scale_factor}")
        if abs(scale_factor - 1.0) > 1e-4:
            for cam in tier2_cameras:
                cam.position = cam.position * scale_factor

    # Run distillation with tier-1 (cubemap) + tier-2 (OmniRoam) views
    # protect_original_count tells the distiller to freeze original Gaussians
    # and only compute tier-2 loss in hole regions
    total_iters = 0
    if len(tier2_images) > 0:
        logger.info(f"[Stage 5] Distilling with {len(cubemap_faces)} tier-1 + "
                    f"{len(tier2_images)} tier-2 views "
                    f"(protecting {initial_gaussian_count} original Gaussians)")

        gaussians = distill_to_gaussians(
            gaussians=gaussians,
            repaired_images=tier2_images,
            cameras=tier2_cameras,
            hole_masks=[v["hole_mask"] for v in selected_views],
            original_images=cubemap_faces,
            original_cameras=cubemap_cameras,
            densify_grad_threshold=config.densify_grad_threshold,
            iters_per_view=config.iters_per_view,
            kf_iters=config.kf_iters,
            protect_original_count=initial_gaussian_count,
        )
        total_iters = config.iters_per_view * len(tier2_images) + config.kf_iters
    else:
        logger.info("[Stage 5] No tier-2 views — distilling with tier-1 only")
        gaussians = distill_to_gaussians(
            gaussians=gaussians,
            repaired_images=cubemap_faces,
            cameras=cubemap_cameras,
            hole_masks=[np.zeros((512, 512), dtype=np.float32)] * len(cubemap_faces),
            iters_per_view=config.iters_per_view,
            kf_iters=config.kf_iters,
        )
        total_iters = config.iters_per_view * len(cubemap_faces) + config.kf_iters

    # Tag any densification-created Gaussians
    post_distill_count = gaussians.get_xyz.shape[0]
    if post_distill_count > pre_seed_count:
        tag_provenance_by_range(
            gaussians, pre_seed_count, post_distill_count, PROVENANCE_OMNIROAM,
        )

    tag_gaussian_provenance(gaussians, initial_gaussian_count)

    # ── Stage 6: Validation & export ────────────────────────────────────
    report("validation", 90)

    # Re-render eval cameras for final coverage
    final_hole_masks, final_azimuths = _render_eval_cameras(
        gaussians, eval_cameras, alpha_threshold=config.hole_mask_threshold,
    )
    final_coverage = compute_coverage(final_hole_masks)
    initial_coverage = compute_coverage(hole_masks)
    final_hole_fraction = 1.0 - final_coverage

    logger.info(f"[Stage 6] Coverage: {initial_coverage:.4f} -> {final_coverage:.4f}")

    # Source-anchor PSNR: render from original viewpoint (front cubemap)
    # and compare to the panorama front face
    source_psnr = 100.0
    if len(cubemap_faces) > 0 and len(cubemap_cameras) > 0:
        anchor_rgb, _mask = render_with_hole_mask(
            gaussians, cubemap_cameras[0], alpha_threshold=0.0,
        )
        source_psnr = compute_psnr(cubemap_faces[0], anchor_rgb)

    anchor_check = check_source_anchor(
        baseline_psnr=source_psnr,  # first iteration baseline = current
        current_psnr=source_psnr,
        floor=config.source_anchor_psnr_floor,
    )

    if not anchor_check.passed:
        logger.warning("[Stage 6] Source anchor check failed — "
                       "refinement may have degraded quality")

    # Export
    report("export", 95)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_gaussians_to_ply(gaussians, output_path)

    final_count = gaussians.get_xyz.shape[0]
    result = _build_result(
        output_path, initial_hole_fraction, final_hole_fraction,
        final_count, total_iters, start_time, source_psnr,
    )

    report("done", 100)
    logger.info(
        f"[refine-v2] Done in {result['total_time']}s. "
        f"Holes: {initial_hole_fraction:.4f} -> {final_hole_fraction:.4f}, "
        f"Gaussians: {initial_gaussian_count} -> {final_count}, "
        f"Source PSNR: {source_psnr:.1f} dB"
    )

    return result
