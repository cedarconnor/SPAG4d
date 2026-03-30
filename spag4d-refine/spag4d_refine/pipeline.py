"""RefinePipeline: end-to-end orchestrator for Gaussian splat refinement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .config import RefineConfig
from .result import RefineResult
from .session import RefineSession
from .gaussian.cloud import GaussianCloud
from .gaussian.provenance import GaussianSource
from .camera.pinhole import CameraSet, PinholeCamera
from .camera.trajectory import generate_orbit_trajectory, generate_preset_trajectory
from .camera.panoramic_extractor import extract_panoramic_view
from .validation.psnr_validator import validate_original_view, check_psnr_regression

logger = logging.getLogger(__name__)


class RefinePipeline:
    """
    Orchestrates the full refinement pipeline.

    Stages:
      0. Load inputs (PLY, panorama, depth)
      1. Camera capture / trajectory generation
      2. Panoramic extraction (for classifier)
      2b. Forward warp (parallax-correct)
      3. Splat rendering along trajectory
      4. Region classification
      5. 3D-aware novel view repair (synthesis)
      6. Depth estimation & Gaussian seeding
      7. Shadow validation & constrained fine-tuning
      8. Validation & pruning
    """

    def __init__(self, config: RefineConfig):
        self.config = config
        self.session = RefineSession()

    def run(
        self,
        cameras: Optional[CameraSet] = None,
        progress_callback: Optional[callable] = None,
    ) -> RefineResult:
        """
        Run the full refinement pipeline.

        Args:
            cameras: Optional pre-defined camera set. If None, generates orbit.
            progress_callback: Optional callback(round_num, stage, pct) for progress updates.

        Returns:
            RefineResult with stats and output path.
        """
        def _progress(round_num: int, stage: str, pct: int):
            if progress_callback is not None:
                progress_callback(round_num, stage, pct)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # === Stage 0: Load inputs ===
        _progress(0, "Loading inputs", 0)
        logger.info("Stage 0: Loading inputs")
        cloud = GaussianCloud.from_ply(self.config.splat_path)
        original_count = len(cloud)
        logger.info(f"  Loaded {original_count:,} Gaussians")

        panorama_rgb = np.array(
            Image.open(self.config.panorama_rgb_path).convert("RGB")
        ).astype(np.float32) / 255.0

        if self.config.panorama_depth_path:
            panorama_depth = np.load(str(self.config.panorama_depth_path))
        else:
            panorama_depth = None

        # === Stage 1: Camera trajectory ===
        logger.info("Stage 1: Camera trajectory")
        if cameras is None:
            cameras = generate_preset_trajectory(
                preset=self.config.camera_preset,
                center=np.zeros(3),
                radius=self.config.orbit_radius,
                n_cameras=self.config.n_cameras,
                vfov_deg=self.config.camera_vfov_deg,
                resolution=self.config.render_resolution,
            )
        logger.info(f"  Using {len(cameras)} cameras")

        # === Synthesis backend (loaded once, reused across cameras) ===
        from .synthesis import get_synthesis_backend
        synth = get_synthesis_backend(self.config)

        # === PSNR baseline (render from panorama center before any refinement) ===
        psnr_before = None
        if panorama_depth is not None:
            ref_camera = PinholeCamera.from_fov(
                vfov_deg=self.config.camera_vfov_deg,
                width=self.config.render_resolution[0],
                height=self.config.render_resolution[1],
                c2w=np.eye(4),
            )
            ref_image = extract_panoramic_view(
                panorama_rgb, panorama_depth, ref_camera,
            ).rgb
            psnr_before = validate_original_view(
                cloud, ref_camera, ref_image, device=self.config.device,
            )
            logger.info(f"  Baseline PSNR: {psnr_before:.2f} dB")

        for round_idx in range(self.config.max_refinement_rounds):
            self.session.begin_round()
            logger.info(f"\n=== Round {self.session.round_number} ===")
            _progress(self.session.round_number, "Starting round", 0)

            gaussians_seeded = 0
            gaussians_promoted = 0
            gaussians_pruned = 0

            # Collect synthesized images + cameras for post-seeding optimization
            synth_targets = []   # [H, W, 3] float32 sRGB
            synth_cameras = []   # PinholeCamera

            for cam_idx, camera in enumerate(cameras):
                cam_pct = int(100 * cam_idx / len(cameras))
                _progress(self.session.round_number, f"Camera {cam_idx + 1}/{len(cameras)}", cam_pct)
                logger.info(f"  Camera {cam_idx + 1}/{len(cameras)}")

                # === Stage 2: Panoramic extraction ===
                pano_result = None
                if panorama_depth is not None:
                    pano_result = extract_panoramic_view(
                        panorama_rgb, panorama_depth, camera,
                    )

                # === Stage 2b: Forward warp ===
                if panorama_depth is not None:
                    from .camera.forward_warper import forward_warp_panorama
                    warp_result = forward_warp_panorama(
                        panorama_rgb, panorama_depth, camera,
                        subsample=self.config.warp_subsample,
                    )
                else:
                    warp_result = None

                # === Stage 3: Splat rendering ===
                from .renderer.gsplat_renderer import GsplatRenderer
                renderer = GsplatRenderer(cloud, device=self.config.device)
                render_result = renderer.render(camera)

                # === Stage 4: Region classification ===
                if warp_result is not None:
                    from .regions.classifier import classify_frame
                    region_map = classify_frame(
                        warp_rgb=warp_result.warped_rgb,
                        warp_valid=warp_result.valid_mask,
                        splat_rgb=render_result.rgb,
                        splat_alpha=render_result.alpha,
                        rgb_error_threshold=self.config.rgb_error_type_a_threshold,
                        alpha_gap_threshold=self.config.alpha_gap_threshold,
                        type_c_alpha_max=self.config.type_c_alpha_max,
                        pano_rgb=pano_result.rgb if pano_result is not None else None,
                        pano_valid=pano_result.valid_mask if pano_result is not None else None,
                        splat_depth=render_result.depth if hasattr(render_result, 'depth') else None,
                        warp_depth=warp_result.warped_depth,
                        depth_disagreement_threshold=self.config.depth_disagreement_threshold,
                    )

                    from .regions.classifier import RegionType
                    gap_mask = (region_map == RegionType.TYPE_C) | (region_map == RegionType.TYPE_A)

                    if not gap_mask.any():
                        logger.info("    No gaps detected, skipping")
                        continue

                    # Save diagnostic images
                    diag_dir = output_dir / "diagnostics"
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    from .renderer.diagnostics import save_diagnostic_bundle
                    save_diagnostic_bundle(
                        splat_rgb=render_result.rgb,
                        warp_rgb=warp_result.warped_rgb,
                        pano_rgb=pano_result.rgb if pano_result is not None else None,
                        region_map=region_map,
                        output_path=diag_dir / f"r{self.session.round_number}_cam{cam_idx}.png",
                    )
                else:
                    # Without depth, use alpha as gap indicator
                    gap_mask = render_result.alpha < self.config.type_c_alpha_max
                    if not gap_mask.any():
                        continue

                # === Stage 5: Synthesis ===
                if hasattr(synth, 'repair') and warp_result is not None:
                    from .synthesis.klein_sharp_repair import RepairInput
                    from .synthesis.depth_visualizer import depth_to_disparity_image

                    # Build dense depth for visualization by filling forward-warp
                    # holes with panoramic extraction depth (same pattern as RGB).
                    # Without this, the depth vis shows striations from sparse warp.
                    filled_depth = warp_result.warped_depth.copy()
                    filled_valid = warp_result.valid_mask.copy()
                    if pano_result is not None:
                        unfilled = ~warp_result.valid_mask
                        filled_depth[unfilled] = pano_result.depth[unfilled]
                        filled_valid[unfilled] = pano_result.valid_mask[unfilled]

                    depth_vis = depth_to_disparity_image(filled_depth, filled_valid)

                    # Build reference image: forward warp (parallax-correct) with
                    # panoramic extraction fallback for unfilled pixels.
                    # The forward warp is sparse, so we need the pano fill to
                    # give Klein a usable reference.
                    reference_rgb = warp_result.warped_rgb.copy()
                    if pano_result is not None:
                        unfilled = ~warp_result.valid_mask
                        reference_rgb[unfilled] = pano_result.rgb[unfilled]

                    # Use appropriate repair interface
                    if self.config.synthesis_backend == "klein-sharp":
                        repair_input = RepairInput(
                            forward_warped_rgb=reference_rgb,
                            broken_splat_render=render_result.rgb,
                            depth_disparity_vis=depth_vis,
                            ref_c2w=np.eye(4),  # Panorama center
                            novel_c2w=camera.c2w,
                            gap_mask=gap_mask,
                        )
                        synthesized = synth.repair(repair_input)
                    else:
                        synthesized = synth.repair(
                            warp_result.warped_rgb, gap_mask
                        )
                    # Save individual diagnostics (including synthesis output)
                    from .renderer.diagnostics import save_individual_diagnostics
                    save_individual_diagnostics(
                        diag_dir=output_dir / "diagnostics",
                        prefix=f"r{self.session.round_number}_cam{cam_idx}",
                        splat_rgb=render_result.rgb,
                        warp_rgb=reference_rgb,
                        pano_rgb=pano_result.rgb if pano_result is not None else None,
                        region_map=region_map if warp_result is not None else None,
                        synthesized=synthesized,
                        depth_vis=depth_vis,
                    )
                    # Collect for post-seeding optimization
                    synth_targets.append(synthesized)
                    synth_cameras.append(camera)
                else:
                    logger.warning("    Synthesis skipped (no backend or no warp)")
                    continue

                # === Stage 6: Depth estimation & seeding ===
                if warp_result is not None:
                    from .seeding.depth_estimator import estimate_and_align_depth
                    aligned_depth = estimate_and_align_depth(
                        synthesized, warp_result.warped_depth,
                        warp_result.valid_mask, gap_mask,
                        device=self.config.device,
                        confidence_decay_pixels=self.config.confidence_decay_pixels,
                    )

                    from .seeding.seeder import seed_shadow_gaussians
                    shadows = seed_shadow_gaussians(
                        aligned_depth, synthesized, gap_mask, camera,
                        shadow_opacity=self.config.shadow_opacity,
                        min_confidence=getattr(self.config, 'min_seed_confidence', 0.3),
                    )

                    gaussians_seeded += len(shadows)
                    logger.info(f"    Camera {cam_idx}: seeded {len(shadows):,} shadow Gaussians")
                    if len(shadows) > 0:
                        cloud = cloud.merge(shadows)

            # === Stage 7: Shadow validation ===
            _progress(self.session.round_number, "Shadow validation", 90)
            from .seeding.shadow_validator import validate_shadow_gaussians
            cloud = validate_shadow_gaussians(
                cloud, list(cameras),
                consistency_threshold=self.config.promotion_consistency_threshold,
            )
            gaussians_promoted = int(np.sum(
                cloud.provenance == GaussianSource.PROMOTED
            ))

            # === Stage 7b: Masked optimization ===
            if synth_targets and self.config.finetune_iterations > 0:
                _progress(self.session.round_number, "Optimizing", 92)
                from .optimization.local_refiner import refine_gaussians
                logger.info(f"  Optimizing against {len(synth_targets)} synthesized views "
                            f"({self.config.finetune_iterations} iters)")
                cloud = refine_gaussians(
                    cloud,
                    target_images=synth_targets,
                    cameras=synth_cameras,
                    iterations=self.config.finetune_iterations,
                    anchor_loss_weight=self.config.anchor_loss_weight,
                    original_lr_scale=self.config.original_gaussian_lr_scale,
                    device=self.config.device,
                )

            # === Stage 8: Validation & pruning ===
            _progress(self.session.round_number, "Pruning", 95)
            from .validation.pruner import prune_gaussians
            n_before_prune = len(cloud)
            cloud = prune_gaussians(cloud)
            gaussians_pruned = n_before_prune - len(cloud)

            # PSNR regression check against original view
            psnr_after = 0.0
            if psnr_before is not None:
                psnr_after = validate_original_view(
                    cloud, ref_camera, ref_image, device=self.config.device,
                )
                if not check_psnr_regression(
                    psnr_before, psnr_after, max_drop=self.config.max_original_psnr_drop,
                ):
                    logger.warning(
                        f"  PSNR hard stop: {psnr_before:.2f} → {psnr_after:.2f} dB "
                        f"(drop > {self.config.max_original_psnr_drop} dB)"
                    )
                    # Save intermediate before breaking
                    intermediate_path = output_dir / f"refined_round{self.session.round_number}.ply"
                    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
                    cloud.to_ply(intermediate_path)
                    self.session.intermediate_ply_paths.append(intermediate_path)
                    self.session.end_round(
                        cameras_processed=len(cameras),
                        gaussians_seeded=gaussians_seeded,
                        gaussians_promoted=gaussians_promoted,
                        gaussians_pruned=gaussians_pruned,
                        original_psnr=psnr_after,
                    )
                    break

            # Save intermediate
            intermediate_path = output_dir / f"refined_round{self.session.round_number}.ply"
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)
            cloud.to_ply(intermediate_path)
            self.session.intermediate_ply_paths.append(intermediate_path)

            self.session.end_round(
                cameras_processed=len(cameras),
                gaussians_seeded=gaussians_seeded,
                gaussians_promoted=gaussians_promoted,
                gaussians_pruned=gaussians_pruned,
                original_psnr=psnr_after,
            )

        # Save final + heatmap
        final_path = output_dir / "refined_final.ply"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        cloud.to_ply(final_path)
        heatmap_path = output_dir / "refined_heatmap.ply"
        cloud.to_heatmap_ply(heatmap_path)
        logger.info(f"Saved heatmap PLY: {heatmap_path}")

        # Compute provenance breakdown
        from .validation.metrics import compute_provenance_breakdown
        provenance = compute_provenance_breakdown(cloud)
        logger.info(f"Provenance breakdown: {provenance}")

        return RefineResult(
            output_path=final_path,
            original_gaussian_count=original_count,
            final_gaussian_count=len(cloud),
            gaussians_added=len(cloud) - original_count,
            gaussians_pruned=sum(s.gaussians_pruned for s in self.session.round_stats),
            original_psnr_before=psnr_before or 0.0,
            original_psnr_after=psnr_after,
            rounds_completed=self.session.round_number,
            round_stats=self.session.round_stats,
            total_elapsed_seconds=self.session.total_elapsed,
            synthesis_backend_used=self.config.synthesis_backend,
            heatmap_path=heatmap_path,
        )
