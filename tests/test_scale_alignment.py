"""Tests for spag4d.refine.scale_alignment."""

import numpy as np
import pytest

from spag4d.refine.scale_alignment import estimate_scale_factor, parse_scale_config


class TestParseScaleConfig:
    def test_none(self):
        assert parse_scale_config("none") is None

    def test_manual(self):
        assert parse_scale_config("manual:2.5") == pytest.approx(2.5)

    def test_manual_default(self):
        assert parse_scale_config("manual:1.0") == pytest.approx(1.0)

    def test_reprojection_returns_sentinel(self):
        assert parse_scale_config("reprojection") == "reprojection"

    def test_none_case_insensitive(self):
        assert parse_scale_config("None") is None

    def test_reprojection_case_insensitive(self):
        assert parse_scale_config("Reprojection") == "reprojection"

    def test_manual_strips_whitespace(self):
        assert parse_scale_config("  manual:3.0  ") == pytest.approx(3.0)

    def test_invalid_manual_value_raises(self):
        with pytest.raises(ValueError, match="Invalid manual scale value"):
            parse_scale_config("manual:not_a_number")

    def test_unrecognised_raises(self):
        with pytest.raises(ValueError, match="Unrecognised scale config"):
            parse_scale_config("auto")


class TestEstimateScaleFactor:
    def test_identity_scale(self):
        rng = np.random.default_rng(0)
        img = rng.random((64, 64, 3)).astype(np.float32)

        def mock_render(scale):
            noise = rng.random((64, 64, 3)).astype(np.float32) * 0.1
            if abs(scale - 1.0) < 0.3:
                return img + noise * (abs(scale - 1.0))
            return noise

        scale = estimate_scale_factor(
            render_fn=mock_render,
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=32,
        )
        assert 0.7 < scale < 1.5

    def test_scaled_scene(self):
        rng = np.random.default_rng(1)
        img = rng.random((64, 64, 3)).astype(np.float32)

        def mock_render(scale):
            noise = rng.random((64, 64, 3)).astype(np.float32) * 0.1
            if abs(scale - 3.0) < 0.5:
                return img + noise * (abs(scale - 3.0) / 3.0)
            return noise

        scale = estimate_scale_factor(
            render_fn=mock_render,
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=32,
        )
        assert 2.0 < scale < 4.5

    def test_fallback_on_no_match(self):
        rng = np.random.default_rng(2)
        img = rng.random((64, 64, 3)).astype(np.float32)

        def mock_render(scale):
            return np.zeros((64, 64, 3), dtype=np.float32)

        scale = estimate_scale_factor(
            render_fn=mock_render,
            omniroam_frame1=img,
            search_range=(0.1, 10.0),
            num_samples=16,
        )
        assert scale == pytest.approx(1.0)

    def test_invalid_search_range_raises(self):
        img = np.ones((8, 8, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="search_range"):
            estimate_scale_factor(
                render_fn=lambda s: img,
                omniroam_frame1=img,
                search_range=(10.0, 0.1),
            )

    def test_zero_reference_falls_back(self):
        """All-zero reference image -> cosine similarity is 0 -> fallback 1.0."""
        ref = np.zeros((8, 8, 3), dtype=np.float32)
        img = np.ones((8, 8, 3), dtype=np.float32)

        scale = estimate_scale_factor(
            render_fn=lambda s: img,
            omniroam_frame1=ref,
            search_range=(0.5, 5.0),
            num_samples=8,
        )
        assert scale == pytest.approx(1.0)

    def test_single_sample(self):
        """num_samples=1 should return the only candidate if similarity is high."""
        img = np.ones((8, 8, 3), dtype=np.float32)
        scale = estimate_scale_factor(
            render_fn=lambda s: img,
            omniroam_frame1=img,
            search_range=(2.0, 2.0001),
            num_samples=1,
        )
        assert 1.9 < scale < 2.1
