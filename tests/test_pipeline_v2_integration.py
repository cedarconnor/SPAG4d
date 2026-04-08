"""Integration tests for the Refine v2 pipeline."""

import numpy as np
import pytest


class TestPipelineImports:
    """Verify all refine-v2 modules import without error."""

    def test_all_modules_import(self):
        from spag4d.refine.omniroam_config import OmniRoamConfig
        from spag4d.refine.omniroam_trajectory import generate_omniroam_trajectory
        from spag4d.refine.gap_analysis import classify_gap_directions, GapReport
        from spag4d.refine.view_selector import extract_perspective_crop
        from spag4d.refine.scale_alignment import estimate_scale_factor
        from spag4d.refine.gap_seeding import seed_gap_gaussians
        from spag4d.refine.validation import compute_psnr, compute_coverage
        from spag4d.refine.pipeline_v2 import refine_splat_v2

    def test_exports(self):
        from spag4d.refine import refine_splat_v2, OmniRoamConfig
        assert callable(refine_splat_v2)
        cfg = OmniRoamConfig()
        assert cfg.enabled is False


class TestGapAnalysisThroughViewSelection:
    """Test gap analysis -> trajectory selection -> view filtering flow."""

    def test_end_to_end_flow(self):
        from spag4d.refine.gap_analysis import classify_gap_directions, select_trajectories
        from spag4d.refine.view_selector import filter_views_by_gap
        from spag4d.refine.omniroam_config import OmniRoamConfig

        # 12 eval views with forward-concentrated gaps
        masks = []
        azimuths = []
        for i in range(12):
            az = i * 30.0
            mask = np.zeros((256, 256), dtype=np.float32)
            if az < 60 or az > 300:
                mask[:] = 0.3
            masks.append(mask)
            azimuths.append(az)

        report = classify_gap_directions(masks, azimuths)
        assert report.avg_hole_fraction > 0

        cfg = OmniRoamConfig(trajectory_mode="auto")
        trajectories = select_trajectories(report, cfg)
        assert len(trajectories) > 0
        assert "forward" in trajectories

        # Simulate candidate views
        views = [
            {"gap_ratio": 0.25, "frame_idx": 0, "direction": 0},
            {"gap_ratio": 0.01, "frame_idx": 1, "direction": 90},
            {"gap_ratio": 0.10, "frame_idx": 2, "direction": 0},
        ]
        selected = filter_views_by_gap(views, min_gap_ratio=0.05, max_views=10)
        assert len(selected) == 2


class TestTrajectoryThroughScaleAlignment:
    """Test trajectory -> scale alignment flow."""

    def test_trajectory_produces_valid_translations(self):
        from spag4d.refine.omniroam_trajectory import generate_omniroam_trajectory
        from spag4d.refine.scale_alignment import estimate_scale_factor

        cam_traj, translations = generate_omniroam_trajectory(preset="forward")
        assert len(translations) == 81
        assert translations[0][0] == 0.0  # starts at origin

        # Mock scale alignment
        frame1_t = translations[1]
        assert np.linalg.norm(frame1_t) > 0  # non-zero offset

        img = np.random.rand(64, 64, 3).astype(np.float32)
        scale = estimate_scale_factor(
            render_fn=lambda s: img,  # always return same image
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=8,
        )
        assert scale > 0


class TestGapSeedingIntegration:
    """Test gap seeding with realistic depth maps."""

    def test_seed_count_reasonable(self):
        from spag4d.refine.gap_seeding import seed_gap_gaussians

        depth = np.random.uniform(2.0, 10.0, (512, 1024)).astype(np.float32)
        gap_mask = np.random.rand(512, 1024) > 0.8  # ~20% gaps

        result = seed_gap_gaussians(depth, gap_mask, stride=4)
        n = result["positions"].shape[0]
        assert 3000 < n < 10000
        assert result["positions"].shape == (n, 3)
        assert result["provenance"] == "gap_seed"


class TestValidationFlow:
    """Test validation metrics integration."""

    def test_psnr_and_coverage_together(self):
        from spag4d.refine.validation import compute_psnr, compute_coverage, check_source_anchor

        # Simulate before/after
        img_before = np.random.rand(64, 64, 3).astype(np.float32)
        img_after = img_before + np.random.rand(64, 64, 3).astype(np.float32) * 0.05
        img_after = np.clip(img_after, 0, 1)

        psnr = compute_psnr(img_before, img_after)
        assert psnr > 20

        masks_before = [np.full((64, 64), 0.3, dtype=np.float32) for _ in range(5)]
        masks_after = [np.full((64, 64), 0.1, dtype=np.float32) for _ in range(5)]
        cov_before = compute_coverage(masks_before)
        cov_after = compute_coverage(masks_after)
        assert cov_after > cov_before

        anchor = check_source_anchor(baseline_psnr=30.0, current_psnr=29.5, floor=25.0)
        assert anchor.passed is True


class TestViewSelectorCropConsistency:
    """Test perspective crop extraction produces valid images."""

    def test_crop_from_gradient_erp(self):
        from spag4d.refine.view_selector import extract_perspective_crop

        # Create gradient ERP image
        erp = np.zeros((480, 960, 3), dtype=np.float32)
        erp[:, :, 0] = np.linspace(0, 1, 960)[None, :]

        crops = {}
        for yaw in [0, 90, 180, 270]:
            crop = extract_perspective_crop(erp, yaw_deg=float(yaw), fov_deg=90.0, size=128)
            assert crop.shape == (128, 128, 3)
            assert crop.dtype == np.float32
            crops[yaw] = crop

        # Different yaws should produce different mean values
        means = {y: crop.mean() for y, crop in crops.items()}
        assert means[0] != pytest.approx(means[180], abs=0.05)
