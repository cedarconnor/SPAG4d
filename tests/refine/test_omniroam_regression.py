"""Regression guard: the ArtiFixer3D addition must be ADDITIVE-ONLY.

The OmniRoam public surface (refine_splat_v2 signature + OmniRoamConfig fields)
must remain byte-stable so backend=omniroam behaves exactly as before.
"""
import inspect

from spag4d.refine import OmniRoamConfig, refine_splat_v2


def test_refine_splat_v2_signature_unchanged():
    params = list(inspect.signature(refine_splat_v2).parameters)
    assert params == [
        "ply_path", "panorama_path", "depth_map",
        "config", "output_path", "progress_callback", "diagnostics_dir",
    ]


def test_omniroam_config_core_fields_present():
    c = OmniRoamConfig()
    # the fields api.py's OmniRoam path constructs must still exist + default
    assert c.enabled is False
    assert c.max_iterations == 3
    assert c.trajectory_mode == "auto"
    assert c.tier2_weight == 0.20
    assert c.upscale_backend == "none"
