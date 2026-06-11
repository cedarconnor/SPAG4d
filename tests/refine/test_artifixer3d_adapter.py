from unittest import mock

from spag4d.refine.artifixer3d_adapter import (
    windows_to_wsl_path,
    validate_artifixer_environment,
)
from spag4d.refine.artifixer3d_config import ArtiFixer3DConfig


def test_windows_to_wsl_path():
    assert windows_to_wsl_path(r"D:\SPAG-4D\work") == "/mnt/d/SPAG-4D/work"


def test_validate_runs_expected_probes():
    cfg = ArtiFixer3DConfig()
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        return mock.Mock(returncode=0, stdout="ok", stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run):
        validate_artifixer_environment(cfg)
    joined = " ".join(" ".join(map(str, c)) for c in calls)
    assert "docker" in joined and "nvidia-smi" in joined  # GPU-in-container gate
    assert cfg.checkpoint in joined                        # checkpoint presence
    assert cfg.wan_mirror in joined                        # local mirror presence


def test_prepare_cmd_has_validated_flags():
    from spag4d.refine.artifixer3d_adapter import build_prepare_cmd
    cmd = " ".join(build_prepare_cmd(ArtiFixer3DConfig(), scene_wsl="/scene", output_root="/scene/prep/s"))
    assert "data_processing.prepare_colmap_artifixer_inputs" in cmd
    assert "--colmap_dir /scene/colmap" in cmd
    assert "--metric_scale 1.0" in cmd
    assert "HF_HUB_OFFLINE=1" in cmd and "--ipc=host" in cmd


def test_inference_cmd_uses_local_model_id_and_offline():
    from spag4d.refine.artifixer3d_adapter import build_inference_cmd
    cmd = " ".join(build_inference_cmd(ArtiFixer3DConfig(), scene_wsl="/scene", split="/scene/prep/s/split.json", save="/scene/prep/s/out"))
    assert "model_eval.run_inference" in cmd
    assert "--evalset reconstructed_colmap" in cmd
    assert "--model_id /data/wan_te" in cmd
    assert "--render_trajectory all_frames" in cmd
    assert "--num_inference_steps 4" in cmd


def test_distill_cmd_uses_wrapper_config_and_base_ckpt():
    from spag4d.refine.artifixer3d_adapter import build_distill_cmd
    cmd = " ".join(build_distill_cmd(ArtiFixer3DConfig(), scene_root="/scene/prep/s", pred_dir="/scene/prep/s/.../pred", base_ckpt="/scene/prep/s/.../ckpt_10000.pt"))
    assert "data_processing.run_artifixer3d" in cmd
    assert "--config_name _artifixer_run" in cmd
    assert "--base_checkpoint" in cmd and "--phases distill,render" in cmd
