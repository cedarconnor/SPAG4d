"""Shared conditioning utilities for synthesis backends."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def compute_relative_transform(
    ref_c2w: np.ndarray,
    novel_c2w: np.ndarray,
) -> Dict[str, float]:
    """
    Compute relative 6DoF transform between reference and novel cameras.

    Returns translation (x, y, z) and rotation (pitch, yaw, roll) in degrees.

    Args:
        ref_c2w: [4, 4] reference camera-to-world
        novel_c2w: [4, 4] novel camera-to-world

    Returns:
        Dict with keys: x, y, z, pitch, yaw, roll
    """
    # Relative transform: novel in reference's frame
    ref_w2c = np.linalg.inv(ref_c2w)
    relative = ref_w2c @ novel_c2w

    # Translation
    tx, ty, tz = relative[:3, 3]

    # Rotation → Euler angles (pitch, yaw, roll)
    R = relative[:3, :3]
    # ZYX Euler angles
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return {
        "x": float(tx),
        "y": float(ty),
        "z": float(tz),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "roll": math.degrees(roll),
    }


def build_repair_prompt(
    transform: Dict[str, float],
    scene_description: str = "indoor scene",
) -> str:
    """
    Build a text prompt for Klein/FLUX synthesis with 6DoF transform.

    Uses the exact format the ml-sharp LoRA was trained on:
      "Referring to the scene in image 1, restore the perspective of the
       scene in image 2. Repair the perspective and missing areas.
       The camera has moved by: {JSON transform}"

    Args:
        transform: Relative transform dict from compute_relative_transform
        scene_description: Brief description of scene content (unused,
            kept for API compatibility)

    Returns:
        Formatted prompt string matching ml-sharp LoRA training format.
    """
    import json

    # Build the camera motion JSON matching the ml-sharp training format
    camera_motion = {
        "x": round(transform["x"], 3),
        "y": round(transform["y"], 3),
        "z": round(transform["z"], 3),
        "pitch": round(transform["pitch"], 1),
        "yaw": round(transform["yaw"], 1),
        "roll": round(transform["roll"], 1),
    }

    return (
        "Referring to the scene in image 1, restore the perspective of "
        "the scene in image 2. Repair the perspective and missing areas. "
        f"The camera has moved by: {json.dumps(camera_motion)}"
    )


def prepare_conditioning_images(
    warped_rgb: np.ndarray,
    splat_rgb: np.ndarray,
    depth_vis: np.ndarray,
    target_size: tuple[int, int] = (1024, 1024),
) -> list[np.ndarray]:
    """
    Prepare the three conditioning images for Klein.

    Args:
        warped_rgb: [H, W, 3] forward-warped RGB (with holes)
        splat_rgb: [H, W, 3] broken splat render
        depth_vis: [H, W, 3] depth disparity visualization

    Returns:
        List of 3 images, each [target_H, target_W, 3] float32 [0, 1]
    """
    from PIL import Image

    images = [warped_rgb, splat_rgb, depth_vis]
    result = []

    for img in images:
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        result.append(np.array(pil_img).astype(np.float32) / 255.0)

    return result
