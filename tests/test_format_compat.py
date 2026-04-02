import numpy as np
import struct
import pytest


def _create_spag4d_ply(path, n=100):
    """Create a minimal SPAG-4D format PLY (no normals)."""
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    rng = np.random.default_rng(42)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for _ in range(n):
            xyz = rng.normal(0, 1, 3).astype(np.float32)
            sh = rng.normal(0, 0.1, 3).astype(np.float32)
            opacity = np.float32(2.0)
            scale = rng.normal(-3, 0.5, 3).astype(np.float32)
            rot = np.array([1, 0, 0, 0], dtype=np.float32)
            f.write(struct.pack('<3f', *xyz))
            f.write(struct.pack('<3f', *sh))
            f.write(struct.pack('<f', opacity))
            f.write(struct.pack('<3f', *scale))
            f.write(struct.pack('<4f', *rot))
    return path


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason="CUDA required")
def test_load_spag4d_ply(tmp_path):
    from spag4d.refine.format_compat import load_gaussians_from_ply
    ply_path = _create_spag4d_ply(tmp_path / "test.ply", n=50)
    gaussians = load_gaussians_from_ply(str(ply_path))
    assert gaussians is not None
    assert gaussians.get_xyz.shape == (50, 3)
    assert gaussians.get_opacity.shape == (50, 1)
    assert gaussians.get_scaling.shape == (50, 3)
    assert gaussians.get_rotation.shape == (50, 4)


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason="CUDA required")
def test_roundtrip(tmp_path):
    from spag4d.refine.format_compat import load_gaussians_from_ply, save_gaussians_to_ply
    ply_path = _create_spag4d_ply(tmp_path / "test.ply", n=50)
    gaussians = load_gaussians_from_ply(str(ply_path))

    out_path = tmp_path / "roundtrip.ply"
    save_gaussians_to_ply(gaussians, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Reload and verify
    gaussians2 = load_gaussians_from_ply(str(out_path))
    import torch
    torch.testing.assert_close(gaussians.get_xyz, gaussians2.get_xyz, atol=1e-5, rtol=1e-5)
