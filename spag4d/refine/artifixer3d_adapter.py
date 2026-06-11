"""ArtiFixer3D WSL2/Docker adapter.

Mirrors ``omniroam_adapter``'s subprocess-orchestration pattern (path conversion,
pre-flight validation, streamed logs) but drives ArtiFixer3D as a sequence of
``docker run`` steps inside WSL — it does NOT import any ArtiFixer library on the
host. Every container contract here was validated in Phase 0/1
(``experiments/artifixer_eval/RESULT.md`` + ``PHASE1.md``).
"""

from __future__ import annotations

import subprocess
from pathlib import PureWindowsPath


# ---------------------------------------------------------------------------
# Path conversion (ported from omniroam_adapter.windows_to_wsl_path)
# ---------------------------------------------------------------------------

def windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to the corresponding WSL mount path.

    >>> windows_to_wsl_path(r"D:\\SPAG-4D\\output")
    '/mnt/d/SPAG-4D/output'
    """
    p = PureWindowsPath(win_path)
    parts = list(p.parts)
    if not parts:
        raise ValueError(f"Cannot convert empty path: {win_path!r}")

    drive = p.drive.rstrip(":\\").lower()
    if not drive:
        raise ValueError(f"Path has no drive letter: {win_path!r}")

    rest = "/".join(parts[1:])
    wsl_path = f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    while "//" in wsl_path:
        wsl_path = wsl_path.replace("//", "/")
    return wsl_path.rstrip("/") if wsl_path != "/" else "/"


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def _configs_dir(config) -> str:
    """WSL path to the 3DGRUT configs dir where the hydra wrappers live."""
    return f"{config.artifixer_repo}/thirdparty/3DGRUT-ArtiFixer/configs"


def validate_artifixer_environment(config) -> None:
    """Pre-flight checks confirming the ArtiFixer3D WSL/Docker stack is ready.

    Probes (each via ``wsl -d <distro> bash -c``):
        (a) GPU visible inside the container (``docker run --gpus all ... nvidia-smi``)
        (b) the 14B checkpoint file exists
        (c) the local Wan2.1 mirror is populated (text_encoder dir + vae weights)
        (d) both hydra wrapper yamls are copied into the repo configs dir

    Raises
    ------
    RuntimeError
        With a fix hint (pointing at setup_wan_mirror.sh / INSTALL.md) on the
        first failed probe.
    """
    distro = config.wsl_distro

    # (a) GPU-in-container gate.
    inner = f"docker run --rm --gpus all {config.docker_image} nvidia-smi -L"
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", inner],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ArtiFixer3D: GPU not visible inside the container "
            f"(image '{config.docker_image}', distro '{distro}'). "
            "Confirm WSL2 + Docker + NVIDIA Container Toolkit per INSTALL.md. "
            f"stderr: {result.stderr.strip()}"
        )

    # (b) checkpoint present.
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", f"test -f {config.checkpoint}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ArtiFixer3D: 14B checkpoint not found at '{config.checkpoint}'. "
            "Place artifixer-14b.pt there (see INSTALL.md)."
        )

    # (c) local Wan2.1 mirror populated.
    mirror_probe = (
        f"test -d {config.wan_mirror}/text_encoder && "
        f"test -f {config.wan_mirror}/vae/diffusion_pytorch_model.safetensors"
    )
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", mirror_probe],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ArtiFixer3D: local Wan2.1 mirror incomplete at '{config.wan_mirror}'. "
            "Run `bash spag4d/refine/artifixer3d_resources/setup_wan_mirror.sh` once "
            "inside WSL (see INSTALL.md)."
        )

    # (d) hydra wrapper configs copied into the repo.
    cfg_dir = _configs_dir(config)
    cfg_probe = (
        f"test -f {cfg_dir}/_artifixer_run.yaml && "
        f"test -f {cfg_dir}/_artifixer_distill.yaml"
    )
    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c", cfg_probe],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ArtiFixer3D: hydra wrapper configs missing in '{cfg_dir}'. "
            "Copy _artifixer_run.yaml + _artifixer_distill.yaml from "
            "spag4d/refine/artifixer3d_resources/ there (see INSTALL.md)."
        )


# ---------------------------------------------------------------------------
# Container-step command builders
#
# Each returns the full ["wsl", "-d", distro, "bash", "-c", inner] list; `inner`
# is a single `docker run` invocation. All container paths below are *inside* the
# container — the host scene dir is mounted at /scene, the repo at
# /workspace/artifixer, /data at /data, and (for caption/export) the vendored
# resources dir at /eval. Every flag matches the Phase 0/1 validated runs
# (see experiments/artifixer_eval/_run_infer.sh, _run_distill.sh, _train_3dgrut.sh,
# RESULT.md, PHASE1.md).
# ---------------------------------------------------------------------------

def _wsl(config, inner: str) -> list:
    return ["wsl", "-d", config.wsl_distro, "bash", "-c", inner]


def _docker_prefix(config, scene_wsl, *, workdir="/workspace/artifixer",
                   extra_mounts=(), extra_envs=()) -> str:
    """The shared `docker run ...` prefix up to (and including) the image + workdir."""
    envs = [
        "-e PYTHONPATH=/workspace/artifixer",
        "-e HF_HUB_OFFLINE=1",
        "-e TRANSFORMERS_OFFLINE=1",
        "-e PYTHONUNBUFFERED=1",
        "-e PYTHONFAULTHANDLER=1",
        f"-e TORCH_HOME={config.torch_cache}",
    ]
    envs += [f"-e {e}" for e in extra_envs]
    mounts = [
        f"-v {config.artifixer_repo}:/workspace/artifixer",
        "-v /data:/data",
        f"-v {scene_wsl}:/scene",
    ]
    mounts += [f"-v {m}" for m in extra_mounts]
    parts = [
        "docker run --rm --gpus all --ipc=host",
        "--ulimit memlock=-1 --ulimit stack=67108864",
        *envs,
        *mounts,
        f"-w {workdir} {config.docker_image}",
    ]
    return " ".join(parts)


def build_prepare_cmd(config, scene_wsl="/scene", output_root="/scene/prep/scene",
                      phases="prepare", selected_image_names_file=None,
                      reconstruction_checkpoint=None) -> list:
    """`data_processing.prepare_colmap_artifixer_inputs` on the bridged COLMAP scene.

    Phase 0 reality: prepare's *reconstruct* phase is broken under hydra 1.3.2
    (the compose() glue), so reconstruction is done out-of-band by
    ``build_train3dgrut_cmd`` and fed back via ``reconstruction_checkpoint`` for
    the render/scale phases. Default ``phases="prepare"`` just lays out the
    3dgrut_input + selected_indices.
    """
    prefix = _docker_prefix(config, scene_wsl)
    args = [
        "python -u -X faulthandler -m data_processing.prepare_colmap_artifixer_inputs",
        "--colmap_dir /scene/colmap",
        f"--output_root {output_root}",
        f"--metric_scale {config.metric_scale}",
        f"--reconstruction_steps {config.recon_steps}",
        f"--text_encoder_model_id {config.wan_mirror}",
        f"--phases {phases}",
    ]
    if selected_image_names_file:
        args.append(f"--selected_image_names_file {selected_image_names_file}")
    if reconstruction_checkpoint:
        args.append(f"--reconstruction_checkpoint {reconstruction_checkpoint}")
    return _wsl(config, prefix + " " + " ".join(args))


def build_train3dgrut_cmd(config, scene_wsl="/scene", scene_root="/scene/prep/scene",
                          scene_name="scene", selected_indices_file=None,
                          num_selected_indices=0) -> list:
    """Native 3DGRUT @hydra.main train bypass (the Phase 0 reconstruct workaround).

    Runs ``train.py --config-name <wrapper>`` inside the 3DGRUT submodule so the
    config composes FLAT (validated in _train_3dgrut.sh). Produces
    ``<scene_root>/3dgrut_runs/<name>/<name>/ours_<recon_steps>/ckpt_<recon_steps>.pt``.
    """
    prefix = _docker_prefix(
        config, scene_wsl,
        workdir="/workspace/artifixer/thirdparty/3DGRUT-ArtiFixer",
    )
    sif = selected_indices_file or f"{scene_root}/selected_indices.json"
    args = [
        f"python -u train.py --config-name {config.distill_config}",
        f"path={scene_root}/3dgrut_input/{scene_name}",
        f"out_dir={scene_root}/3dgrut_runs",
        f"selected_indices_file={sif}",
        f"experiment_name={scene_name}",
        f"num_selected_indices={num_selected_indices}",
        "test_last=False export_ingp.enabled=False",
        f"n_iterations={config.recon_steps}",
        f"'checkpoint.iterations=[{config.recon_steps}]'",
    ]
    return _wsl(config, prefix + " " + " ".join(args))


def build_caption_cmd(config, scene_wsl="/scene", resources_wsl="/eval",
                      scene_root="/scene/prep/scene", scene_name="scene") -> list:
    """Run the vendored make_caption.py (UMT5-only, VLM bypassed) to emit caption.h5.

    The resources dir (containing make_caption.py) is mounted at /eval; scene paths
    are passed via SCENE_ROOT/SCENE_NAME env (make_caption.py reads them).
    """
    prefix = _docker_prefix(
        config, scene_wsl,
        extra_mounts=(f"{resources_wsl}:/eval",),
        extra_envs=(f"SCENE_ROOT={scene_root}", f"SCENE_NAME={scene_name}"),
    )
    return _wsl(config, prefix + " python -u -X faulthandler /eval/make_caption.py")


def build_inference_cmd(config, scene_wsl="/scene", split="/scene/prep/scene/split.json",
                        save="/scene/prep/scene/artifixer_out",
                        render_trajectory="all_frames") -> list:
    """`model_eval.run_inference` — the 14B ArtiFixer DiT 2D correction pass."""
    prefix = _docker_prefix(config, scene_wsl)
    args = [
        "python -u -X faulthandler -m model_eval.run_inference",
        "--evalset reconstructed_colmap",
        f"--model_id {config.wan_mirror}",
        f"--checkpoint_pt {config.checkpoint}",
        f"--save_dir {save}",
        f"--split_path {split}",
        f"--render_trajectory {render_trajectory}",
        f"--num_inference_steps {config.num_inference_steps}",
    ]
    return _wsl(config, prefix + " " + " ".join(args))


def build_distill_cmd(config, scene_root="/scene/prep/scene", pred_dir="",
                      base_ckpt="", scene_wsl="/scene", phases="distill,render") -> list:
    """`data_processing.run_artifixer3d` — distill the 2D pred frames into a 3D cloud."""
    prefix = _docker_prefix(config, scene_wsl)
    args = [
        "python -u -X faulthandler -m data_processing.run_artifixer3d",
        f"--scene_root {scene_root}",
        f"--artifixer_frames_dir {pred_dir}",
        f"--base_checkpoint {base_ckpt}",
        f"--config_name {config.distill_config}",
        f"--artifixer3d_steps {config.distill_steps}",
        f"--phases {phases}",
    ]
    return _wsl(config, prefix + " " + " ".join(args))


def build_export_cmd(config, scene_wsl="/scene", resources_wsl="/eval",
                     ckpt="", out_ply="") -> list:
    """Run the vendored export_ply.py: distilled 3DGRUT ckpt -> standard 3DGS PLY."""
    prefix = _docker_prefix(
        config, scene_wsl,
        workdir="/workspace/artifixer/thirdparty/3DGRUT-ArtiFixer",
        extra_mounts=(f"{resources_wsl}:/eval",),
    )
    return _wsl(config, prefix + f" python -u /eval/export_ply.py {ckpt} {out_ply}")
