# tests/test_ply_compat.py
"""
Verify PLY files are compatible with target viewers.
"""

import numpy as np
import torch
import pytest
import tempfile
from pathlib import Path

from spag4d.ply_writer import save_ply_gsplat, load_ply_gaussians


def _create_test_gaussians(n: int) -> dict:
    """Create synthetic Gaussian data for testing."""
    return {
        'means': torch.randn(n, 3),
        'scales': torch.rand(n, 3) * 0.1 + 0.01,
        'quats': torch.randn(n, 4),
        'colors': torch.rand(n, 3),
        'opacities': torch.rand(n, 1) * 0.5 + 0.5,
    }


class TestPlySchema:
    """Verify PLY file structure matches viewer expectations."""

    def test_ply_has_required_properties(self, tmp_path):
        """PLY should have all properties required by SuperSplat/gsplat."""
        from plyfile import PlyData

        gaussians = _create_test_gaussians(100)

        ply_path = tmp_path / "test.ply"
        save_ply_gsplat(gaussians, str(ply_path), sh_degree=0)

        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']

        required = [
            'x', 'y', 'z',
            'nx', 'ny', 'nz',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ]

        for prop in required:
            assert prop in vertex.data.dtype.names, f"Missing required property: {prop}"

    def test_ply_sh_degree_3(self, tmp_path):
        """SH degree 3 should add f_rest_* properties."""
        from plyfile import PlyData

        gaussians = _create_test_gaussians(100)

        ply_path = tmp_path / "test_sh3.ply"
        save_ply_gsplat(gaussians, str(ply_path), sh_degree=3)

        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']

        # SH degree 3 has 45 rest coefficients (48 total - 3 DC)
        num_rest = (3 + 1) ** 2 * 3 - 3  # = 45

        for i in range(num_rest):
            assert f'f_rest_{i}' in vertex.data.dtype.names, f"Missing f_rest_{i}"

    def test_ply_roundtrip(self, tmp_path):
        """Save and load should preserve data."""
        gaussians = _create_test_gaussians(100)

        ply_path = tmp_path / "roundtrip.ply"
        save_ply_gsplat(gaussians, str(ply_path))

        loaded = load_ply_gaussians(str(ply_path))

        assert loaded['means'].shape == gaussians['means'].shape
        assert loaded['colors'].shape == gaussians['colors'].shape

    def test_ply_nonempty_file(self, tmp_path):
        """PLY output should be a non-empty file."""
        gaussians = _create_test_gaussians(1000)

        ply_path = tmp_path / "test.ply"
        save_ply_gsplat(gaussians, str(ply_path))

        assert ply_path.exists()
        assert ply_path.stat().st_size > 0

    def test_ply_colors_in_valid_range(self, tmp_path):
        """SH DC coefficients should be finite after linearRGB->sRGB conversion."""
        from plyfile import PlyData

        gaussians = _create_test_gaussians(100)
        # Ensure colors are in [0, 1] (linearRGB range)
        gaussians['colors'] = gaussians['colors'].clamp(0, 1)

        ply_path = tmp_path / "color_test.ply"
        save_ply_gsplat(gaussians, str(ply_path), sh_degree=0)

        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']

        for ch in ('f_dc_0', 'f_dc_1', 'f_dc_2'):
            vals = vertex[ch]
            assert np.all(np.isfinite(vals)), f"{ch} contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
