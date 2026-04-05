"""SeedVR2 video upscaling adapter — WSL2 subprocess wrapper.

Uses ComfyUI-SeedVR2_VideoUpscaler's standalone inference_cli.py
to upscale OmniRoam-generated ERP video before perspective crop extraction.
"""

import logging
import re
import subprocess
from typing import Callable, List, Optional

from .omniroam_adapter import windows_to_wsl_path

logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(r"(\d+)/(\d+)")


def validate_seedvr2_environment(config) -> None:
    """Check that SeedVR2 CLI is available in the WSL2 conda env.

    Raises RuntimeError if inference_cli.py is not found.
    """
    distro = config.wsl_distro
    install_dir = config.seedvr2_install_dir

    result = subprocess.run(
        ["wsl", "-d", distro, "bash", "-c",
         f"source ~/miniconda3/etc/profile.d/conda.sh && "
         f"conda activate omniroam && "
         f"test -f {install_dir}/inference_cli.py"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SeedVR2 not found at '{install_dir}/inference_cli.py' "
            f"in WSL distro '{distro}'. "
            f"Install with: wsl bash -c 'git clone "
            f"https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "
            f"{install_dir}'"
        )
    logger.info("SeedVR2 environment validated OK")


def _build_seedvr2_args(
    input_path: str,
    output_path: str,
    config,
) -> list:
    """Build the CLI argument list for inference_cli.py."""
    return [
        input_path,
        "--output", output_path,
        "--dit_model", config.seedvr2_model,
        "--resolution", str(config.seedvr2_target_resolution),
        "--batch_size", str(config.seedvr2_batch_size),
        "--color_correction", config.seedvr2_color_correction,
        "--blocks_to_swap", str(config.seedvr2_block_swap),
        "--cuda_device", "0",
        "--video_backend", "opencv",
    ]


def run_seedvr2_upscale(
    video_path: str,
    output_path: str,
    config,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Upscale a video using SeedVR2 inside WSL2.

    Args:
        video_path: Windows path to input video (OmniRoam generated.mp4).
        output_path: Windows path for the upscaled output video.
        config: OmniRoamConfig with seedvr2_* fields.
        progress_callback: Optional (current_batch, total_batches) callback.

    Returns:
        output_path on success.

    Raises:
        RuntimeError if SeedVR2 process fails.
    """
    distro = config.wsl_distro
    install_dir = config.seedvr2_install_dir

    wsl_input = windows_to_wsl_path(video_path)
    wsl_output = windows_to_wsl_path(output_path)

    cli_args = _build_seedvr2_args(wsl_input, wsl_output, config)
    args_str = " ".join(cli_args)

    inner_cmd = (
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate omniroam && "
        f"cd {install_dir} && "
        f"python inference_cli.py {args_str}"
    )

    cmd = ["wsl", "-d", distro, "bash", "-c", inner_cmd]
    logger.info(f"Running SeedVR2 upscale: {config.seedvr2_target_resolution}p, "
                f"model={config.seedvr2_model}")

    log_lines: List[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    for line in proc.stdout:
        log_lines.append(line)
        if len(log_lines) > 20:
            log_lines.pop(0)

        if progress_callback is not None:
            match = _PROGRESS_RE.search(line)
            if match:
                progress_callback(int(match.group(1)), int(match.group(2)))

    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"SeedVR2 failed with exit code {returncode}.\n"
            f"Last output:\n{''.join(log_lines)}"
        )

    logger.info("SeedVR2 upscale complete")
    return output_path
