"""Tests for spag4d.refine.gap_analysis."""

import numpy as np
import pytest

from spag4d.refine.gap_analysis import (
    GapReport,
    classify_gap_directions,
    select_trajectories,
    _azimuth_to_direction,
)


# ---------------------------------------------------------------------------
# Unit tests for _azimuth_to_direction
# ---------------------------------------------------------------------------

class TestAzimuthToDirection:
    def test_zero_is_forward(self):
        assert _azimuth_to_direction(0.0) == "forward"

    def test_360_wraps_to_forward(self):
        assert _azimuth_to_direction(360.0) == "forward"

    def test_30_is_forward(self):
        assert _azimuth_to_direction(30.0) == "forward"

    def test_315_is_forward(self):
        assert _azimuth_to_direction(315.0) == "forward"

    def test_359_is_forward(self):
        assert _azimuth_to_direction(359.0) == "forward"

    def test_90_is_right(self):
        assert _azimuth_to_direction(90.0) == "right"

    def test_45_is_right(self):
        assert _azimuth_to_direction(45.0) == "right"

    def test_134_is_right(self):
        assert _azimuth_to_direction(134.9) == "right"

    def test_180_is_backward(self):
        assert _azimuth_to_direction(180.0) == "backward"

    def test_135_is_backward(self):
        assert _azimuth_to_direction(135.0) == "backward"

    def test_270_is_left(self):
        assert _azimuth_to_direction(270.0) == "left"

    def test_225_is_left(self):
        assert _azimuth_to_direction(225.0) == "left"

    def test_314_is_left(self):
        assert _azimuth_to_direction(314.9) == "left"


# ---------------------------------------------------------------------------
# Tests for classify_gap_directions
# ---------------------------------------------------------------------------

class TestClassifyGapDirections:

    def test_forward_gaps(self):
        """High hole fractions at forward azimuths → forward is worst direction."""
        masks = []
        azimuths = []
        for azi_deg in range(0, 360, 30):
            mask = np.zeros((512, 512), dtype=np.float32)
            if abs(azi_deg) < 45 or abs(azi_deg - 360) < 45:
                mask[:, :] = 0.8
            else:
                mask[:, :] = 0.01
            masks.append(mask)
            azimuths.append(float(azi_deg))

        report = classify_gap_directions(masks, azimuths)
        assert isinstance(report, GapReport)
        assert report.worst_direction == "forward"
        assert report.avg_hole_fraction > 0

    def test_bilateral_gaps(self):
        """High hole fractions at right/left azimuths → both appear in trajectories."""
        masks = []
        azimuths = []
        for azi_deg in range(0, 360, 30):
            mask = np.zeros((512, 512), dtype=np.float32)
            if 60 <= azi_deg <= 120 or 240 <= azi_deg <= 300:
                mask[:, :] = 0.6
            else:
                mask[:, :] = 0.01
            masks.append(mask)
            azimuths.append(float(azi_deg))

        report = classify_gap_directions(masks, azimuths)
        assert (
            "left" in report.recommended_trajectories
            or "right" in report.recommended_trajectories
        )

    def test_no_gaps(self):
        """Very low hole fractions everywhere → converged, no trajectories recommended."""
        masks = [np.full((512, 512), 0.01, dtype=np.float32) for _ in range(12)]
        azimuths = [float(i * 30) for i in range(12)]
        report = classify_gap_directions(masks, azimuths, min_hole_fraction=0.05)
        assert len(report.recommended_trajectories) == 0
        assert report.converged is True

    def test_returns_gap_report_instance(self):
        masks = [np.zeros((64, 64), dtype=np.float32)]
        report = classify_gap_directions(masks, [0.0])
        assert isinstance(report, GapReport)

    def test_empty_input(self):
        """Empty input should return a zeroed, converged GapReport."""
        report = classify_gap_directions([], [])
        assert report.avg_hole_fraction == 0.0
        assert report.converged is True
        assert report.recommended_trajectories == []

    def test_worst_direction_is_correct(self):
        """Manually construct known fractions and verify worst direction."""
        # All viewpoints at azimuth 180 (backward) with high hole fraction.
        masks = [np.full((64, 64), 0.9, dtype=np.float32) for _ in range(4)]
        azimuths = [180.0] * 4
        report = classify_gap_directions(masks, azimuths, min_hole_fraction=0.05)
        assert report.worst_direction == "backward"
        assert "backward" in report.recommended_trajectories

    def test_recommended_sorted_by_severity(self):
        """Trajectories in recommended list should be ordered worst-first."""
        masks = []
        azimuths = []
        # forward ≈ 0.8, right ≈ 0.5, others ≈ 0.01
        for azi, frac in [(0.0, 0.8), (90.0, 0.5), (180.0, 0.01), (270.0, 0.01)]:
            mask = np.full((64, 64), frac, dtype=np.float32)
            masks.append(mask)
            azimuths.append(azi)

        report = classify_gap_directions(masks, azimuths, min_hole_fraction=0.05)
        recs = report.recommended_trajectories
        assert recs[0] == "forward"
        assert recs[1] == "right"

    def test_per_direction_fractions_keys(self):
        """per_direction_fractions should always have all four cardinal keys."""
        masks = [np.zeros((32, 32), dtype=np.float32)]
        report = classify_gap_directions(masks, [45.0])
        assert set(report.per_direction_fractions.keys()) == {
            "forward", "right", "backward", "left"
        }

    def test_converged_false_when_above_threshold(self):
        masks = [np.full((64, 64), 0.9, dtype=np.float32)]
        azimuths = [0.0]
        report = classify_gap_directions(masks, azimuths, min_hole_fraction=0.05)
        assert report.converged is False

    def test_custom_min_hole_fraction(self):
        """High threshold means even large gaps converge."""
        masks = [np.full((64, 64), 0.3, dtype=np.float32)]
        azimuths = [0.0]
        report = classify_gap_directions(masks, azimuths, min_hole_fraction=0.5)
        assert report.converged is True
        assert report.recommended_trajectories == []


# ---------------------------------------------------------------------------
# Tests for select_trajectories
# ---------------------------------------------------------------------------

class TestSelectTrajectories:

    def test_auto_selects_from_gap_report(self):
        report = GapReport(
            avg_hole_fraction=0.15,
            per_direction_fractions={
                "forward": 0.3,
                "right": 0.2,
                "backward": 0.02,
                "left": 0.02,
            },
            worst_direction="forward",
            recommended_trajectories=["forward", "right"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode="auto")
        result = select_trajectories(report, cfg)
        assert "forward" in result
        assert "right" in result
        assert "backward" not in result

    def test_all_mode(self):
        report = GapReport(
            avg_hole_fraction=0.15,
            per_direction_fractions={},
            worst_direction="forward",
            recommended_trajectories=["forward"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode="all")
        result = select_trajectories(report, cfg)
        assert set(result) == {"forward", "backward", "left", "right"}

    def test_explicit_list(self):
        report = GapReport(
            avg_hole_fraction=0.15,
            per_direction_fractions={},
            worst_direction="forward",
            recommended_trajectories=["forward"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode=["forward", "s_curve"])
        result = select_trajectories(report, cfg)
        assert result == ["forward", "s_curve"]

    def test_auto_empty_when_converged(self):
        """Auto mode with converged report returns empty list."""
        report = GapReport(
            avg_hole_fraction=0.01,
            per_direction_fractions={
                "forward": 0.01,
                "right": 0.01,
                "backward": 0.01,
                "left": 0.01,
            },
            worst_direction="forward",
            recommended_trajectories=[],
            converged=True,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode="auto")
        result = select_trajectories(report, cfg)
        assert result == []

    def test_explicit_list_ignores_report(self):
        """An explicit list should be returned as-is regardless of report content."""
        report = GapReport(
            avg_hole_fraction=0.9,
            per_direction_fractions={},
            worst_direction="backward",
            recommended_trajectories=["forward", "right", "backward", "left"],
            converged=False,
        )
        from spag4d.refine.omniroam_config import OmniRoamConfig
        cfg = OmniRoamConfig(trajectory_mode=["left"])
        result = select_trajectories(report, cfg)
        assert result == ["left"]
