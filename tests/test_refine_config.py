from spag4d.refine.config import RefineConfig


def test_config_defaults():
    cfg = RefineConfig()
    assert cfg.camera_fov == 60.0
    assert cfg.max_iterations == 3
    assert cfg.convergence_threshold == 0.02
    assert cfg.finetune_steps == 500
    assert cfg.distill_iterations == 3000
    assert len(cfg.translation_fracs) == 3


def test_config_override():
    cfg = RefineConfig(max_iterations=5, finetune_steps=200)
    assert cfg.max_iterations == 5
    assert cfg.finetune_steps == 200
    assert cfg.camera_fov == 60.0
