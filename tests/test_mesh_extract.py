"""Tests for spag4d.refine.mesh_extract — conditioning mesh from depth."""

import numpy as np
import pytest


def test_extract_mesh_from_synthetic_depth():
    from spag4d.refine.mesh_extract import extract_conditioning_mesh

    h, w = 64, 128
    depth = np.ones((h, w), dtype=np.float32) * 3.0
    panorama = np.random.rand(h, w, 3).astype(np.float32)

    mesh = extract_conditioning_mesh(depth, panorama, simplify_ratio=0.5)
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_extract_mesh_too_few_points():
    from spag4d.refine.mesh_extract import extract_conditioning_mesh

    # Very small depth map with mostly zeros
    depth = np.zeros((4, 4), dtype=np.float32)
    panorama = np.zeros((4, 4, 3), dtype=np.float32)

    mesh = extract_conditioning_mesh(depth, panorama)
    assert mesh is None


def test_render_mesh_none():
    from spag4d.refine.mesh_extract import render_mesh
    from spag4d.refine.camera_rig import CameraPose

    cam = CameraPose(
        position=np.zeros(3), look_at=np.array([0, 0, -1.0]),
        up=np.array([0, 1.0, 0]), fov_deg=60.0, width=64, height=64,
    )
    result = render_mesh(None, cam, resolution=(64, 64))
    assert result.shape == (64, 64, 3)
    assert np.allclose(result, 0.5)
