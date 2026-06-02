# tests/refine/geometric/test_validity_mask.py
import numpy as np
from spag4d.refine.geometric.masks import (
    sky_mask_hsv,
    specular_mask,
    build_validity_mask,
)


def test_sky_mask_upper_hemisphere_high_value():
    H, W = 64, 128
    image = np.zeros((H, W, 3), dtype=np.uint8)
    # Upper quarter: bright near-white pixels (sky-like)
    image[: H // 4, :] = [220, 230, 240]
    mask = sky_mask_hsv(image)
    # Upper region should mostly be sky
    assert mask[: H // 4, :].mean() > 0.5
    # Lower region should not be sky
    assert mask[H // 2 :, :].mean() < 0.1


def test_specular_mask_bright_low_texture():
    H, W = 32, 32
    image = np.full((H, W, 3), 250, dtype=np.uint8)  # near-saturated bright
    mask = specular_mask(image)
    assert mask.mean() > 0.5


def test_build_validity_mask_excludes_low_alpha():
    H, W = 64, 128
    alpha = np.ones((H, W), dtype=np.float32)
    alpha[:, :10] = 0.5  # low alpha on left strip
    depth = np.full((H, W), 5.0, dtype=np.float32)
    sky = np.zeros((H, W), dtype=bool)
    spec = np.zeros((H, W), dtype=bool)
    mask = build_validity_mask(alpha, depth, sky, spec, alpha_confident=0.9)
    assert mask[:, :10].any() == False
    assert mask[:, 20:].mean() > 0.9


def test_build_validity_mask_excludes_sky():
    H, W = 64, 128
    alpha = np.ones((H, W), dtype=np.float32)
    depth = np.full((H, W), 5.0, dtype=np.float32)
    sky = np.zeros((H, W), dtype=bool)
    sky[:10, :] = True
    spec = np.zeros((H, W), dtype=bool)
    mask = build_validity_mask(alpha, depth, sky, spec)
    assert mask[:10, :].any() == False
