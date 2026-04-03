"""Tests for OmniRoam trajectory generator."""

import numpy as np
import pytest
import torch
from pathlib import Path

from spag4d.refine.omniroam_trajectory import generate_omniroam_trajectory

PRESETS = ["forward", "backward", "left", "right", "s_curve", "loop"]
SNAPSHOT_DIR = Path(__file__).parent / "data" / "omniroam_trajectory_snapshots"


class TestTrajectoryShape:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_cam_traj_shape(self, preset):
        cam_traj, translations = generate_omniroam_trajectory(preset=preset)
        assert cam_traj.shape == (21, 12), f"Expected (21, 12), got {cam_traj.shape}"

    @pytest.mark.parametrize("preset", PRESETS)
    def test_cam_traj_dtype(self, preset):
        cam_traj, _ = generate_omniroam_trajectory(preset=preset)
        assert cam_traj.dtype == torch.float32

    @pytest.mark.parametrize("preset", PRESETS)
    def test_translations_length(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        assert len(translations) == 81

    @pytest.mark.parametrize("preset", PRESETS)
    def test_translations_element_shape(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        for i, t in enumerate(translations):
            assert t.shape == (3,), f"Frame {i}: expected shape (3,), got {t.shape}"

    @pytest.mark.parametrize("preset", PRESETS)
    def test_cam_traj_is_tensor(self, preset):
        cam_traj, _ = generate_omniroam_trajectory(preset=preset)
        assert isinstance(cam_traj, torch.Tensor)

    @pytest.mark.parametrize("preset", PRESETS)
    def test_translations_is_list(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        assert isinstance(translations, list)

    @pytest.mark.parametrize("preset", PRESETS)
    def test_translations_are_numpy_arrays(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        for t in translations:
            assert isinstance(t, np.ndarray)


class TestTrajectoryGeometry:
    def test_frame0_is_origin_forward(self):
        _, translations = generate_omniroam_trajectory(preset="forward")
        np.testing.assert_allclose(translations[0], [0.0, 0.0, 0.0])

    @pytest.mark.parametrize("preset", PRESETS)
    def test_frame0_is_always_origin(self, preset):
        _, translations = generate_omniroam_trajectory(preset=preset)
        np.testing.assert_allclose(translations[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_forward_moves_positive_x(self):
        _, translations = generate_omniroam_trajectory(preset="forward")
        # Last frame should be at positive X
        assert translations[-1][0] > 0.0, "forward should move in +X"
        # Y should always be 0
        for t in translations:
            assert t[1] == pytest.approx(0.0)
        # Z should always be 0
        for t in translations:
            assert t[2] == pytest.approx(0.0)

    def test_backward_moves_negative_x(self):
        _, translations = generate_omniroam_trajectory(preset="backward")
        assert translations[-1][0] < 0.0, "backward should move in -X"
        for t in translations:
            assert t[1] == pytest.approx(0.0)
            assert t[2] == pytest.approx(0.0)

    def test_right_moves_positive_z(self):
        _, translations = generate_omniroam_trajectory(preset="right")
        assert translations[-1][2] > 0.0, "right should move in +Z"
        for t in translations:
            assert t[0] == pytest.approx(0.0)
            assert t[1] == pytest.approx(0.0)

    def test_left_moves_negative_z(self):
        _, translations = generate_omniroam_trajectory(preset="left")
        assert translations[-1][2] < 0.0, "left should move in -Z"
        for t in translations:
            assert t[0] == pytest.approx(0.0)
            assert t[1] == pytest.approx(0.0)

    def test_s_curve_y_always_zero(self):
        _, translations = generate_omniroam_trajectory(preset="s_curve")
        for i, t in enumerate(translations):
            assert t[1] == pytest.approx(0.0), f"s_curve frame {i}: Y={t[1]} should be 0"

    def test_s_curve_nonzero_z_variation(self):
        _, translations = generate_omniroam_trajectory(preset="s_curve")
        z_values = [t[2] for t in translations]
        assert max(z_values) > 0.5, "s_curve should have significant Z variation"
        assert min(z_values) < -0.5, "s_curve should have negative Z values"

    def test_s_curve_x_monotone_increasing(self):
        _, translations = generate_omniroam_trajectory(preset="s_curve")
        x_values = [t[0] for t in translations]
        for i in range(1, len(x_values)):
            assert x_values[i] >= x_values[i - 1], f"s_curve X should be non-decreasing at frame {i}"

    def test_loop_returns_near_origin(self):
        _, translations = generate_omniroam_trajectory(preset="loop")
        # Frame 80 corresponds to theta = 2pi, so cos(2pi)=1 -> x=R*(1-1)=0, sin(2pi)=0 -> z=0
        t80 = translations[80]
        np.testing.assert_allclose(t80[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(t80[2], 0.0, atol=1e-5)

    def test_loop_y_always_zero(self):
        _, translations = generate_omniroam_trajectory(preset="loop")
        for i, t in enumerate(translations):
            assert t[1] == pytest.approx(0.0), f"loop frame {i}: Y={t[1]} should be 0"

    def test_linear_presets_uniform_step(self):
        """Forward/backward/left/right should have uniform spacing."""
        for preset in ("forward", "backward", "left", "right"):
            _, translations = generate_omniroam_trajectory(preset=preset)
            diffs = [np.linalg.norm(translations[i + 1] - translations[i])
                     for i in range(len(translations) - 1)]
            ref = diffs[0]
            for i, d in enumerate(diffs):
                assert d == pytest.approx(ref, abs=1e-6), (
                    f"{preset}: non-uniform step at frame {i}: {d} vs {ref}"
                )

    def test_forward_total_displacement(self):
        """Total displacement for forward = step_m/4 * (num_frames-1)."""
        step_m = 0.25
        num_video_frames = 81
        _, translations = generate_omniroam_trajectory(
            preset="forward", step_m=step_m, num_video_frames=num_video_frames
        )
        expected_x = (step_m / 4.0) * (num_video_frames - 1)
        np.testing.assert_allclose(translations[-1][0], expected_x, rtol=1e-5)

    def test_loop_midpoint(self):
        """At frame 40 (theta=pi), loop should be at x=2R, z~0."""
        R = 1.5
        _, translations = generate_omniroam_trajectory(preset="loop", loop_radius_m=R)
        t40 = translations[40]
        theta = 2.0 * np.pi * 40 / 80.0  # = pi
        expected_x = R * (1.0 - np.cos(theta))  # = R*(1-(-1)) = 2R
        expected_z = R * np.sin(theta)  # = 0
        np.testing.assert_allclose(t40[0], expected_x, atol=1e-5)
        np.testing.assert_allclose(t40[2], expected_z, atol=1e-5)


class TestKeyframeSubsampling:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_keyframe_count(self, preset):
        cam_traj, _ = generate_omniroam_trajectory(preset=preset)
        assert cam_traj.shape[0] == 21

    @pytest.mark.parametrize("preset", PRESETS)
    def test_keyframes_are_every_4th_frame(self, preset):
        cam_traj, translations = generate_omniroam_trajectory(preset=preset)
        for k in range(21):
            j = 4 * k
            t_expected = translations[j]
            # Last 3 elements of each 12-element keyframe row are translation
            M = cam_traj[k].numpy().reshape(3, 4)
            t_actual = M[:, 3]
            np.testing.assert_allclose(
                t_actual, t_expected, atol=1e-6,
                err_msg=f"{preset}: keyframe {k} (frame {j}) translation mismatch"
            )

    @pytest.mark.parametrize("preset", PRESETS)
    def test_rotation_part_is_identity(self, preset):
        cam_traj, _ = generate_omniroam_trajectory(preset=preset)
        identity = np.eye(3)
        for k in range(21):
            M = cam_traj[k].numpy().reshape(3, 4)
            R = M[:, :3]
            np.testing.assert_allclose(
                R, identity, atol=1e-6,
                err_msg=f"{preset}: keyframe {k} rotation is not identity"
            )

    @pytest.mark.parametrize("preset", PRESETS)
    def test_first_keyframe_is_frame_0(self, preset):
        cam_traj, translations = generate_omniroam_trajectory(preset=preset)
        M = cam_traj[0].numpy().reshape(3, 4)
        np.testing.assert_allclose(M[:, 3], translations[0], atol=1e-6)

    @pytest.mark.parametrize("preset", PRESETS)
    def test_last_keyframe_is_frame_80(self, preset):
        cam_traj, translations = generate_omniroam_trajectory(preset=preset)
        M = cam_traj[20].numpy().reshape(3, 4)
        np.testing.assert_allclose(M[:, 3], translations[80], atol=1e-6)

    @pytest.mark.parametrize("preset", PRESETS)
    def test_keyframe_matrix_shape(self, preset):
        """Each row is a flattened [I|t] 3x4 matrix."""
        cam_traj, _ = generate_omniroam_trajectory(preset=preset)
        for k in range(21):
            row = cam_traj[k].numpy()
            assert row.shape == (12,), f"{preset}: keyframe {k} row shape {row.shape}"

    def test_custom_num_keyframes(self):
        cam_traj, _ = generate_omniroam_trajectory(
            preset="forward", num_keyframes=11, num_video_frames=41
        )
        assert cam_traj.shape == (11, 12)

    def test_custom_num_video_frames(self):
        cam_traj, translations = generate_omniroam_trajectory(
            preset="forward", num_video_frames=41, num_keyframes=11
        )
        assert len(translations) == 41


class TestInvalidPreset:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            generate_omniroam_trajectory(preset="bogus")

    def test_error_message_includes_preset_name(self):
        with pytest.raises(ValueError, match="bogus"):
            generate_omniroam_trajectory(preset="bogus")

    @pytest.mark.parametrize("bad_preset", ["Forward", "LOOP", "spiral", "", "none"])
    def test_various_invalid_presets(self, bad_preset):
        with pytest.raises(ValueError):
            generate_omniroam_trajectory(preset=bad_preset)


class TestSnapshotParity:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_matches_snapshot(self, preset):
        snapshot_path = SNAPSHOT_DIR / f"{preset}.npy"
        if not snapshot_path.exists():
            pytest.skip(f"Snapshot not found: {snapshot_path}")

        expected = np.load(snapshot_path)
        cam_traj, _ = generate_omniroam_trajectory(preset=preset)
        actual = cam_traj.numpy()

        np.testing.assert_allclose(
            actual, expected, rtol=1e-5, atol=1e-6,
            err_msg=f"Snapshot mismatch for preset '{preset}'"
        )

    @pytest.mark.parametrize("preset", PRESETS)
    def test_snapshot_shape(self, preset):
        snapshot_path = SNAPSHOT_DIR / f"{preset}.npy"
        if not snapshot_path.exists():
            pytest.skip(f"Snapshot not found: {snapshot_path}")

        data = np.load(snapshot_path)
        assert data.shape == (21, 12), f"{preset} snapshot shape: {data.shape}"

    @pytest.mark.parametrize("preset", PRESETS)
    def test_snapshot_dtype(self, preset):
        snapshot_path = SNAPSHOT_DIR / f"{preset}.npy"
        if not snapshot_path.exists():
            pytest.skip(f"Snapshot not found: {snapshot_path}")

        data = np.load(snapshot_path)
        assert data.dtype == np.float32, f"{preset} snapshot dtype: {data.dtype}"
