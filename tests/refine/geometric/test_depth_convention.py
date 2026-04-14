# tests/refine/geometric/test_depth_convention.py
import numpy as np
import pytest
from spag4d.refine.geometric.depth_convention import (
    is_nearer_than_rendered,
    radial_to_z,
    erp_pixel_to_ray,
)


def test_is_nearer_returns_true_when_candidate_strictly_in_front():
    candidate_z = np.array([1.8, 5.0])
    rendered_z = np.array([2.0, 5.0])
    result = is_nearer_than_rendered(candidate_z, rendered_z, margin_ratio=0.02)
    assert result[0] == True   # 1.8 < 2.0 - 0.04
    assert result[1] == False  # 5.0 is not in front of 5.0


def test_is_nearer_with_local_depth_scale():
    candidate_z = np.array([0.9])
    rendered_z = np.array([1.0])
    local_scale = np.array([10.0])
    # margin = 0.02 * 10 = 0.2; threshold = 1.0 - 0.2 = 0.8
    result = is_nearer_than_rendered(candidate_z, rendered_z, margin_ratio=0.02, local_depth_scale=local_scale)
    assert result[0] == False  # 0.9 > 0.8, not nearer


def test_radial_to_z_at_equator():
    # At equator (lat=0), radial == z
    radial = np.array([5.0])
    lat = np.array([0.0])
    lon = np.array([0.0])
    z = radial_to_z(radial, lat, lon)
    np.testing.assert_allclose(z, radial, rtol=1e-5)


def test_radial_to_z_at_pole():
    # At lat=90 deg (straight up), z approaches 0 (forward is horizontal)
    radial = np.array([5.0])
    lat = np.array([np.pi / 2])
    lon = np.array([0.0])
    z = radial_to_z(radial, lat, lon)
    np.testing.assert_allclose(z, np.array([0.0]), atol=1e-5)


def test_erp_pixel_to_ray_center():
    H, W = 480, 960
    u, v = np.array([W // 2]), np.array([H // 2])
    rays = erp_pixel_to_ray(u, v, H, W)
    # Center pixel → forward ray (0, 0, 1) in camera space
    np.testing.assert_allclose(rays[0], [0.0, 0.0, 1.0], atol=1e-5)
