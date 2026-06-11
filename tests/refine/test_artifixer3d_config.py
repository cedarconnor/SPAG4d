from spag4d.refine.artifixer3d_config import ArtiFixer3DConfig


def test_defaults_match_validated_environment():
    c = ArtiFixer3DConfig()
    assert c.enabled is False
    assert c.docker_image == "artifixer:cuda12"
    assert c.wsl_distro == "Ubuntu"
    assert c.checkpoint == "/data/artifixer-checkpoints/artifixer-14b.pt"
    assert c.wan_mirror == "/data/wan_te"
    assert c.artifixer_repo == "/home/cedarconnor/ArtiFixer"
    assert c.num_inference_steps == 4
    assert c.distill_steps == 30000
    assert c.recon_steps == 10000
    # bridge / anchor-novel
    assert c.num_directions == 16
    assert c.translation_fracs == (0.08, 0.20, 0.40)
    assert c.render_resolution == 640
    assert c.fov_deg == 70.0
    assert 0.0 < c.anchor_parallax_quantile < 1.0
