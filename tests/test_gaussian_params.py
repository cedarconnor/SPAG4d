# tests/test_gaussian_params.py
"""
Tests for Phase 2: Gaussian parameterization.

Covers gaussian_params.py (numpy reference) and the oriented-Gaussian
integration in gaussian_converter.py.
"""

import numpy as np
import pytest
import torch


# ── gaussian_params (numpy reference) ────────────────────────────────────────

class TestEstimateNormals:
    def test_output_shape(self):
        depth = np.ones((64, 128), dtype=np.float32) * 5.0
        from spag4d.gaussian_params import estimate_normals_from_erp_depth
        normals, conf = estimate_normals_from_erp_depth(depth, H=64, W=128)
        assert normals.shape == (64, 128, 3)
        assert conf.shape == (64, 128)

    def test_normals_unit_length(self):
        from spag4d.gaussian_params import estimate_normals_from_erp_depth
        depth = np.random.uniform(2.0, 10.0, (32, 64)).astype(np.float32)
        normals, _ = estimate_normals_from_erp_depth(depth, H=32, W=64)
        mags = np.linalg.norm(normals, axis=-1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-5)

    def test_normals_inward_on_smooth_sphere(self):
        """On a constant-depth sphere (poles excluded), >90% of normals point inward."""
        from spag4d.gaussian_params import estimate_normals_from_erp_depth
        H, W = 64, 128
        depth = np.full((H, W), 5.0, dtype=np.float32)
        normals, _ = estimate_normals_from_erp_depth(depth, H=H, W=W)

        # Build positions
        rows = np.arange(H)[:, None] * np.ones((1, W))
        cols = np.ones((H, 1)) * np.arange(W)[None, :]
        phi   = rows / H * np.pi
        theta = (1.0 - cols / W) * 2.0 * np.pi
        x =  depth * np.sin(phi) * np.cos(theta)
        y =  depth * np.cos(phi)
        z = -depth * np.sin(phi) * np.sin(theta)
        pos = np.stack([x, y, z], axis=-1)

        # Exclude pole rows (top/bottom 3) where cross-product is degenerate
        pole_rows = 3
        inner = slice(pole_rows, H - pole_rows)
        dots = np.sum(normals[inner] * pos[inner], axis=-1)
        inward_pct = (dots < 0).mean()
        assert inward_pct > 0.90, f"Only {inward_pct*100:.1f}% inward on smooth sphere"

    def test_confidence_low_at_edges(self):
        """Depth discontinuity → low confidence at the step edge."""
        from spag4d.gaussian_params import estimate_normals_from_erp_depth
        H, W = 64, 128
        depth = np.full((H, W), 5.0, dtype=np.float32)
        depth[:, W // 2:] = 10.0          # step edge at column 64
        _, conf = estimate_normals_from_erp_depth(depth, H=H, W=W)

        edge_conf    = float(conf[:, W//2 - 2: W//2 + 2].mean())
        non_edge_conf = float(conf[:, :W//4].mean())
        assert edge_conf < non_edge_conf, (
            f"Edge conf {edge_conf:.3f} should be < non-edge {non_edge_conf:.3f}"
        )

    def test_confidence_high_on_flat_surface(self):
        """Constant-depth map → zero depth gradient → confidence close to 1."""
        from spag4d.gaussian_params import estimate_normals_from_erp_depth
        depth = np.full((32, 64), 5.0, dtype=np.float32)
        _, conf = estimate_normals_from_erp_depth(depth, H=32, W=64)
        assert conf.mean() > 0.95

    def test_strided_depth(self):
        """Strided depth map (h < H) should still produce correct shapes."""
        from spag4d.gaussian_params import estimate_normals_from_erp_depth
        depth = np.ones((32, 64), dtype=np.float32) * 3.0   # stride=2, orig=64×128
        normals, conf = estimate_normals_from_erp_depth(depth, H=64, W=128)
        assert normals.shape == (32, 64, 3)
        assert conf.shape == (32, 64)


class TestComputeGaussianScales:
    def test_output_shape(self):
        from spag4d.gaussian_params import compute_gaussian_scales
        depth = np.random.uniform(1.0, 5.0, (32, 64)).astype(np.float32)
        s_iso, s_h, s_v = compute_gaussian_scales(depth, stride=2, H=64, W=128)
        assert s_iso.shape == (32, 64)
        assert s_h.shape == (32, 64)
        assert s_v.shape == (32, 64)

    def test_scale_proportional_to_depth(self):
        """Scales at depth=10 should be ~2× scales at depth=5."""
        from spag4d.gaussian_params import compute_gaussian_scales
        d5  = np.full((16, 32), 5.0, dtype=np.float32)
        d10 = np.full((16, 32), 10.0, dtype=np.float32)
        s_iso5,  _, _ = compute_gaussian_scales(d5,  stride=1, H=16, W=32)
        s_iso10, _, _ = compute_gaussian_scales(d10, stride=1, H=16, W=32)
        ratio = s_iso10 / s_iso5
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05)

    def test_all_scales_positive(self):
        from spag4d.gaussian_params import compute_gaussian_scales
        depth = np.random.uniform(0.5, 20.0, (16, 32)).astype(np.float32)
        s_iso, s_h, s_v = compute_gaussian_scales(depth, stride=2, H=32, W=64)
        assert (s_iso > 0).all()
        assert (s_h > 0).all()
        assert (s_v > 0).all()


class TestNormalToCovariance:
    def test_output_shapes(self):
        from spag4d.gaussian_params import normal_to_covariance
        q, ls = normal_to_covariance(np.array([0., 0., 1.]), 0.05, 0.04)
        assert q.shape == (4,)
        assert ls.shape == (3,)

    def test_quaternion_unit_length(self):
        from spag4d.gaussian_params import normal_to_covariance
        for n in [
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0.577, 0.577, 0.577],
        ]:
            q, _ = normal_to_covariance(np.array(n, dtype=np.float32), 0.05, 0.04)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)

    def test_disc_thinness(self):
        """log_scales[2] (normal axis) should be < log_scales[0] and [1]."""
        from spag4d.gaussian_params import normal_to_covariance
        _, ls = normal_to_covariance(np.array([0., 0., 1.]), 0.05, 0.04, thickness_ratio=0.1)
        assert ls[2] < ls[0]
        assert ls[2] < ls[1]

    def test_thickness_ratio_applied(self):
        """scale_n ≈ thickness_ratio * min(scale_h, scale_v)."""
        from spag4d.gaussian_params import normal_to_covariance
        sh, sv, tr = 0.05, 0.04, 0.1
        _, ls = normal_to_covariance(np.array([0., 0., 1.]), sh, sv, tr)
        expected_sn = tr * min(sh, sv)
        np.testing.assert_allclose(np.exp(ls[2]), expected_sn, rtol=0.01)


# ── gaussian_converter (torch integration) ────────────────────────────────────

class TestOrientedGaussians:
    @pytest.fixture
    def grid(self):
        from spag4d.spherical_grid import create_spherical_grid
        return create_spherical_grid(64, 128, torch.device('cpu'), stride=2)

    def test_oriented_matches_shape(self, grid):
        """oriented_gaussians=True produces same output shape as False."""
        from spag4d.gaussian_converter import equirect_to_gaussians
        image = torch.randint(0, 255, (64, 128, 3), dtype=torch.uint8)
        depth = torch.full((64, 128), 5.0)

        out_oriented = equirect_to_gaussians(image, depth, grid, oriented_gaussians=True)
        out_spherical = equirect_to_gaussians(image, depth, grid, oriented_gaussians=False)

        for key in ('means', 'scales', 'quats', 'opacities'):
            assert out_oriented[key].shape == out_spherical[key].shape, key

    def test_oriented_produces_different_quats(self, grid):
        """Depth-derived normals should give different quaternions than pure spherical."""
        from spag4d.gaussian_converter import equirect_to_gaussians
        image = torch.randint(0, 255, (64, 128, 3), dtype=torch.uint8)
        # Depth that produces a non-spherical surface (sloped plane)
        rows = torch.arange(32, dtype=torch.float32)[:, None].expand(32, 64)
        depth = 2.0 + rows * 0.1   # depth increases with row → sloped floor

        out_ori = equirect_to_gaussians(image, depth, grid, oriented_gaussians=True)
        out_sph = equirect_to_gaussians(image, depth, grid, oriented_gaussians=False)

        # Quaternions should differ for sloped surfaces
        quat_diff = (out_ori['quats'] - out_sph['quats']).abs().mean().item()
        assert quat_diff > 0.01, f"Quats nearly identical (diff={quat_diff:.5f})"

    def test_normal_confidence_modulates_opacity(self, grid):
        """Step-edge depth → confidence < 1 → opacity < default_opacity at edge pixels."""
        from spag4d.gaussian_converter import equirect_to_gaussians
        image = torch.randint(0, 255, (64, 128, 3), dtype=torch.uint8)
        # Sharp step at column 32 (strided: col 16)
        depth = torch.full((64, 128), 5.0)
        depth[:, 64:] = 15.0

        out = equirect_to_gaussians(image, depth, grid, oriented_gaussians=True,
                                     default_opacity=0.95)
        # Max opacity should be ≤ 0.95 * 1.2 clamped to 0.99, min > 0
        assert out['opacities'].max().item() <= 0.99
        assert out['opacities'].min().item() > 0.0

    def test_quats_normalized(self, grid):
        """All output quaternions should be unit-length."""
        from spag4d.gaussian_converter import equirect_to_gaussians
        image = torch.randint(0, 255, (64, 128, 3), dtype=torch.uint8)
        depth = torch.rand(64, 128) * 8.0 + 2.0

        out = equirect_to_gaussians(image, depth, grid, oriented_gaussians=True)
        qnorm = out['quats'].norm(dim=-1)
        torch.testing.assert_close(qnorm, torch.ones_like(qnorm), atol=1e-4, rtol=0)

    def test_ply_has_quaternion_fields(self, tmp_path, grid):
        """PLY output from oriented Gaussians still has rot_0..3 and scale_0..2."""
        import plyfile
        from spag4d.gaussian_converter import equirect_to_gaussians
        from spag4d.ply_writer import save_ply_gsplat

        image = torch.randint(0, 255, (64, 128, 3), dtype=torch.uint8)
        depth = torch.full((64, 128), 5.0)
        out = equirect_to_gaussians(image, depth, grid, oriented_gaussians=True)

        ply_path = str(tmp_path / "test.ply")
        save_ply_gsplat(out, ply_path)

        ply = plyfile.PlyData.read(ply_path)
        names = ply['vertex'].data.dtype.names
        for field in ('rot_0', 'rot_1', 'rot_2', 'rot_3',
                       'scale_0', 'scale_1', 'scale_2'):
            assert field in names, f"Missing PLY field: {field}"
