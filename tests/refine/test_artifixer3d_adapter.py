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
