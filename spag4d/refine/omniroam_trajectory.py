"""OmniRoam trajectory generation — reimplements make_cam_traj_from_preset_refspace()."""

import numpy as np
import torch


def generate_omniroam_trajectory(
    preset: str = "forward",
    step_m: float = 0.25,
    amp_m: float = 1.6,
    loop_radius_m: float = 1.5,
    num_video_frames: int = 81,
    num_keyframes: int = 21,
) -> tuple:
    """Generate OmniRoam-compatible camera trajectory.

    Reimplements OmniRoam's ``make_cam_traj_from_preset_refspace()`` so that
    SPAG-4D has both the model-conditioning tensor and per-frame translations
    without a live OmniRoam install.

    Args:
        preset: One of "forward", "backward", "left", "right", "s_curve", "loop".
        step_m: Linear step size in metres (used by directional and s_curve presets).
        amp_m: Lateral amplitude in metres (s_curve only).
        loop_radius_m: Circle radius in metres (loop only).
        num_video_frames: Total number of per-frame translation entries.
        num_keyframes: Number of keyframe matrices to emit (every 4th frame).

    Returns:
        cam_traj: ``(num_keyframes, 12)`` float32 tensor of flattened [I|t] matrices.
        translations: List of ``num_video_frames`` (3,) numpy arrays (world-space XYZ).

    Raises:
        ValueError: If *preset* is not one of the recognised strings.
    """
    translations = []

    if preset in ("forward", "backward", "left", "right"):
        dir_map = {
            "forward":  np.array([+1.0, 0.0, 0.0]),
            "backward": np.array([-1.0, 0.0, 0.0]),
            "right":    np.array([0.0, 0.0, +1.0]),
            "left":     np.array([0.0, 0.0, -1.0]),
        }
        d = dir_map[preset] * (step_m / 4.0)
        p = np.zeros(3)
        for _ in range(num_video_frames):
            translations.append(p.copy())
            p = p + d

    elif preset == "s_curve":
        for i in range(num_video_frames):
            x = (step_m / 4.0) * i
            z = amp_m * np.sin(2.0 * np.pi * i / 80.0)
            translations.append(np.array([x, 0.0, z]))

    elif preset == "loop":
        R = loop_radius_m
        for i in range(num_video_frames):
            theta = 2.0 * np.pi * i / 80.0
            x = R * (1.0 - np.cos(theta))
            z = R * np.sin(theta)
            translations.append(np.array([x, 0.0, z]))

    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    keyframes = []
    for k in range(num_keyframes):
        j = 4 * k
        t = translations[j]
        M = np.concatenate([np.eye(3), t.reshape(3, 1)], axis=1)
        keyframes.append(M.reshape(-1))

    cam_traj = torch.from_numpy(np.stack(keyframes).astype(np.float32))
    return cam_traj, translations
