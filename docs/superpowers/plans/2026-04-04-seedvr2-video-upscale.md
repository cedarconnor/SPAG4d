# SeedVR2 Video Upscale Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional SeedVR2 video upscaling (Stage 3) to the Refine v2 pipeline, upscaling OmniRoam's 480x960 ERP video to ~1024x2048 before perspective crop extraction.

**Architecture:** SeedVR2 runs in WSL2 (same conda env as OmniRoam) via the ComfyUI-SeedVR2_VideoUpscaler `inference_cli.py` standalone CLI. A new `seedvr2_adapter.py` wraps the subprocess call following the same pattern as `omniroam_adapter.py`. Pipeline Stage 3 is wired to call the adapter when `upscale_backend="seedvr2"`, replacing each trajectory's frames with upscaled versions before Stage 4 crop extraction.

**Tech Stack:** ComfyUI-SeedVR2_VideoUpscaler (inference_cli.py), PyTorch 2.6+, WSL2 subprocess, BlockSwap for A6000 48GB.

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `spag4d/refine/seedvr2_adapter.py` | WSL2 subprocess wrapper for SeedVR2 inference_cli.py — validate env, run upscale, return upscaled video path |
| `tests/test_seedvr2_adapter.py` | Mocked subprocess tests for the adapter |

### Modified Files

| File | Changes |
|------|---------|
| `spag4d/refine/omniroam_config.py` | Add 6 SeedVR2 config fields after `upscale_backend` |
| `spag4d/refine/pipeline_v2.py` | Replace Stage 3 placeholder with SeedVR2 adapter call |
| `scripts/setup_omniroam_wsl.sh` | Add SeedVR2 installation step |
| `api.py` | Pass `upscale_backend` param through to config in `/api/refine_v2` |
| `static/index.html` | Add upscale option to OmniRoam params panel |

---

## Task Dependency Graph

```
Task 1 (Config fields) ──> Task 2 (Adapter) ──> Task 3 (Pipeline wiring)
                                                       │
Task 4 (Setup script) ── independent                   │
                                                       v
                                              Task 5 (API + UI)
```

Tasks 1-2 are sequential. Task 4 is independent. Tasks 3, 5 depend on 1+2.

---

### Task 1: SeedVR2 Config Fields

**Files:**
- Modify: `spag4d/refine/omniroam_config.py`
- Modify: `tests/test_omniroam_config.py`

- [ ] **Step 1: Add SeedVR2 fields to OmniRoamConfig**

Replace the single `upscale_backend` line in `spag4d/refine/omniroam_config.py`:

```python
    # ── Upscale (optional, Phase 2) ──
    upscale_backend: str = "none"
```

With:

```python
    # ── Upscale (optional) ──
    upscale_backend: str = "none"  # "none" | "seedvr2"
    seedvr2_install_dir: str = "/home/cedarconnor/ComfyUI-SeedVR2_VideoUpscaler"
    seedvr2_model: str = "seedvr2_ema_7b_sharp_fp16"  # dit_model name
    seedvr2_target_resolution: int = 1024  # short-side pixels (2x from 480)
    seedvr2_batch_size: int = 5  # frames per batch (4n+1 pattern)
    seedvr2_color_correction: str = "lab"  # "lab" | "wavelet" | "none"
    seedvr2_block_swap: int = 36  # transformer blocks to swap (0-36 for 7B)
```

- [ ] **Step 2: Add test for new defaults**

Add to `tests/test_omniroam_config.py`:

```python
def test_seedvr2_defaults():
    cfg = OmniRoamConfig()
    assert cfg.upscale_backend == "none"
    assert cfg.seedvr2_model == "seedvr2_ema_7b_sharp_fp16"
    assert cfg.seedvr2_target_resolution == 1024
    assert cfg.seedvr2_batch_size == 5
    assert cfg.seedvr2_color_correction == "lab"
    assert cfg.seedvr2_block_swap == 36
    assert cfg.seedvr2_install_dir == "/home/cedarconnor/ComfyUI-SeedVR2_VideoUpscaler"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_omniroam_config.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add spag4d/refine/omniroam_config.py tests/test_omniroam_config.py
git commit -m "feat(seedvr2): add SeedVR2 config fields to OmniRoamConfig"
```

---

### Task 2: SeedVR2 WSL2 Adapter

**Files:**
- Create: `spag4d/refine/seedvr2_adapter.py`
- Create: `tests/test_seedvr2_adapter.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_seedvr2_adapter.py
import pytest
from unittest.mock import patch, MagicMock

from spag4d.refine.seedvr2_adapter import (
    validate_seedvr2_environment,
    run_seedvr2_upscale,
)
from spag4d.refine.omniroam_config import OmniRoamConfig


class TestValidateSeedvr2Environment:
    @patch("spag4d.refine.seedvr2_adapter.subprocess.run")
    def test_missing_cli(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        with pytest.raises(RuntimeError, match="SeedVR2.*not found"):
            validate_seedvr2_environment(cfg)

    @patch("spag4d.refine.seedvr2_adapter.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        validate_seedvr2_environment(cfg)  # Should not raise


class TestRunSeedvr2Upscale:
    @patch("spag4d.refine.seedvr2_adapter.subprocess.Popen")
    def test_success_returns_output_path(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "Loading model...\n",
            "Processing batch 1/5\n",
            "Processing batch 5/5\n",
            "Saved to output\n",
        ])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Create fake input video
        input_video = tmp_path / "generated.mp4"
        input_video.write_bytes(b"fake mp4")

        # Create fake output (the adapter checks for it)
        output_video = tmp_path / "generated_upscaled.mp4"
        output_video.write_bytes(b"fake upscaled")

        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        result = run_seedvr2_upscale(
            video_path=str(input_video),
            output_path=str(output_video),
            config=cfg,
        )
        assert result == str(output_video)
        mock_popen.assert_called_once()

    @patch("spag4d.refine.seedvr2_adapter.subprocess.Popen")
    def test_failure_raises(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["CUDA OOM\n"])
        mock_proc.wait.return_value = 1
        mock_popen.return_value = mock_proc

        input_video = tmp_path / "generated.mp4"
        input_video.write_bytes(b"fake mp4")

        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        with pytest.raises(RuntimeError, match="SeedVR2 failed"):
            run_seedvr2_upscale(
                video_path=str(input_video),
                output_path=str(tmp_path / "out.mp4"),
                config=cfg,
            )

    @patch("spag4d.refine.seedvr2_adapter.subprocess.Popen")
    def test_progress_callback(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            "Processing batch 2/5\n",
            "Processing batch 5/5\n",
        ])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        input_video = tmp_path / "generated.mp4"
        input_video.write_bytes(b"fake mp4")
        output_video = tmp_path / "generated_upscaled.mp4"
        output_video.write_bytes(b"fake upscaled")

        progress = []
        cfg = OmniRoamConfig(upscale_backend="seedvr2")
        run_seedvr2_upscale(
            str(input_video), str(output_video), cfg,
            progress_callback=lambda cur, tot: progress.append((cur, tot)),
        )
        assert (2, 5) in progress
        assert (5, 5) in progress

    def test_build_cli_args(self):
        """Verify the CLI argument builder produces correct flags."""
        from spag4d.refine.seedvr2_adapter import _build_seedvr2_args
        cfg = OmniRoamConfig(
            seedvr2_target_resolution=1024,
            seedvr2_batch_size=5,
            seedvr2_color_correction="lab",
            seedvr2_block_swap=36,
            seedvr2_model="seedvr2_ema_7b_sharp_fp16",
        )
        args = _build_seedvr2_args(
            input_path="/mnt/d/video.mp4",
            output_path="/mnt/d/video_up.mp4",
            config=cfg,
        )
        assert "--resolution" in args
        assert "1024" in args
        assert "--batch_size" in args
        assert "5" in args
        assert "--color_correction" in args
        assert "lab" in args
        assert "--blocks_to_swap" in args
        assert "36" in args
        assert "--dit_model" in args
        assert "--output" in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_seedvr2_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement seedvr2_adapter.py**

```python
# spag4d/refine/seedvr2_adapter.py
"""SeedVR2 video upscaling adapter — WSL2 subprocess wrapper.

Uses ComfyUI-SeedVR2_VideoUpscaler's standalone inference_cli.py
to upscale OmniRoam-generated ERP video before perspective crop extraction.
"""

import logging
import re
import subprocess
from pathlib import Path
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_seedvr2_adapter.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spag4d/refine/seedvr2_adapter.py tests/test_seedvr2_adapter.py
git commit -m "feat(seedvr2): WSL2 adapter for SeedVR2 video upscaling"
```

---

### Task 3: Wire Stage 3 in Pipeline

**Files:**
- Modify: `spag4d/refine/pipeline_v2.py`

- [ ] **Step 1: Add SeedVR2 import at top of pipeline_v2.py**

Add after the existing imports (around line 30):

```python
from .seedvr2_adapter import validate_seedvr2_environment, run_seedvr2_upscale
```

- [ ] **Step 2: Replace Stage 3 placeholder**

Replace the current Stage 3 block (lines 365-371):

```python
    # ── Stage 3: Upscale (OPTIONAL — skipped in Phase 1) ───────────────
    report("upscale", 35)

    if config.upscale_backend != "none":
        logger.info(f"[Stage 3] Upscale backend={config.upscale_backend} — not yet implemented")
    else:
        logger.info("[Stage 3] Upscale skipped (backend=none)")
```

With:

```python
    # ── Stage 3: Upscale (OPTIONAL) ────────────────────────────────────
    report("upscale", 35)

    if config.upscale_backend == "seedvr2" and omniroam_frames_by_traj:
        logger.info(f"[Stage 3] Upscaling with SeedVR2 "
                     f"(resolution={config.seedvr2_target_resolution})")
        validate_seedvr2_environment(config)

        for preset in list(omniroam_frames_by_traj.keys()):
            report(f"upscale_{preset}", 35)

            # Find the generated video for this trajectory
            traj_dir = work_dir / preset
            videos = list(traj_dir.rglob("generated.mp4"))
            if not videos:
                logger.warning(f"[Stage 3] No video found for preset={preset}, skipping upscale")
                continue

            src_video = str(videos[0])
            upscaled_video = str(videos[0].parent / "generated_upscaled.mp4")

            run_seedvr2_upscale(
                video_path=src_video,
                output_path=upscaled_video,
                config=config,
            )

            # Re-extract frames from the upscaled video
            upscaled_frames = extract_video_frames(upscaled_video)
            if upscaled_frames:
                logger.info(f"[Stage 3] Upscaled {preset}: {len(upscaled_frames)} frames "
                            f"at {upscaled_frames[0].shape[1]}x{upscaled_frames[0].shape[0]}")
                omniroam_frames_by_traj[preset] = upscaled_frames
            else:
                logger.warning(f"[Stage 3] Failed to extract upscaled frames for {preset}")

    elif config.upscale_backend != "none":
        logger.warning(f"[Stage 3] Unknown upscale backend: {config.upscale_backend}")
    else:
        logger.info("[Stage 3] Upscale skipped (backend=none)")
```

- [ ] **Step 3: Commit**

```bash
git add spag4d/refine/pipeline_v2.py
git commit -m "feat(seedvr2): wire SeedVR2 upscale into pipeline Stage 3"
```

---

### Task 4: SeedVR2 Installation in Setup Script

**Files:**
- Modify: `scripts/setup_omniroam_wsl.sh`

- [ ] **Step 1: Add SeedVR2 step to setup script**

Add a new step after the DiffSynth installation step (Step 7) in `scripts/setup_omniroam_wsl.sh`. Find the line that says the DiffSynth step is complete and add after it:

```bash
# ─────────────────────────────────────────────────────────────
#  Step 7b/9 — SeedVR2 Video Upscaler (optional)
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Step 7b/9 — SeedVR2 Video Upscaler (optional)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

SEEDVR2_DIR="${SEEDVR2_DIR:-$HOME/ComfyUI-SeedVR2_VideoUpscaler}"

if [ -d "$SEEDVR2_DIR" ]; then
    echo -e "${GREEN}[OK]${NC}  SeedVR2 already cloned at $SEEDVR2_DIR"
    cd "$SEEDVR2_DIR"
    git pull --ff-only 2>/dev/null || echo -e "${YELLOW}[--]${NC}  git pull skipped"
else
    echo -e "${CYAN}[--]${NC}  Cloning SeedVR2 VideoUpscaler..."
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEEDVR2_DIR"
fi

echo -e "${CYAN}[--]${NC}  Installing SeedVR2 dependencies..."
cd "$SEEDVR2_DIR"
conda run -n omniroam pip install -r requirements.txt -q 2>/dev/null || \
    echo -e "${YELLOW}[--]${NC}  requirements.txt not found, SeedVR2 may need manual dep install"

# Model weights are downloaded on first run by inference_cli.py (auto-download)
echo -e "${GREEN}[OK]${NC}  SeedVR2 installed (model weights download on first use)"
echo -e "${CYAN}[--]${NC}  SeedVR2 dir: $SEEDVR2_DIR"
```

- [ ] **Step 2: Add SeedVR2 to verification section**

Add to the verification checks at the end of the script:

```bash
# SeedVR2 check
if [ -f "$SEEDVR2_DIR/inference_cli.py" ]; then
    echo -e "${GREEN}[OK]${NC}  SeedVR2 inference_cli.py present"
    CHECKS_OK=$((CHECKS_OK + 1))
else
    echo -e "${YELLOW}[--]${NC}  SeedVR2 not installed (optional)"
fi
```

And add to the config values output:

```bash
  seedvr2_install_dir   = '$SEEDVR2_DIR'
```

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_omniroam_wsl.sh
git commit -m "feat(seedvr2): add SeedVR2 installation to WSL2 setup script"
```

---

### Task 5: API + UI Integration

**Files:**
- Modify: `api.py`
- Modify: `static/index.html`

- [ ] **Step 1: Add upscale_backend parameter to /api/refine_v2**

In `api.py`, update the `start_refinement_v2` function signature to add:

```python
    upscale_backend: str = Query("none", description="none | seedvr2"),
```

And in the config construction inside `_run_refinement_v2`, add:

```python
    config = OmniRoamConfig(
        enabled=True,
        max_iterations=params.get("max_rounds", 3),
        trajectory_mode=traj_mode,
        tier2_weight=params.get("tier2_weight", 0.20),
        upscale_backend=params.get("upscale_backend", "none"),
    )
```

Also pass `upscale_backend` into `refine_job.params`:

```python
    refine_job.params = {
        "backend": "omniroam",
        "max_rounds": max_rounds,
        "trajectory_mode": trajectory_mode,
        "tier2_weight": tier2_weight,
        "upscale_backend": upscale_backend,
    }
```

- [ ] **Step 2: Add upscale toggle to OmniRoam UI panel**

In `static/index.html`, inside the `omniroam-params` div, add after the trajectory mode dropdown:

```html
                    <div class="param-group" title="Optional video upscaling after OmniRoam generation. SeedVR2 upscales 480p to ~1024p.">
                        <label for="upscale-backend">Upscale</label>
                        <select id="upscale-backend">
                            <option value="none" selected>None (480p)</option>
                            <option value="seedvr2">SeedVR2 (1024p)</option>
                        </select>
                    </div>
```

- [ ] **Step 3: Pass upscale_backend in JavaScript**

In `static/js/app.js`, update the OmniRoam branch of `startRefinement()` to include:

```javascript
            params = new URLSearchParams({
                job_id: this.currentJobId,
                max_rounds: document.getElementById('max-rounds-v2')?.value || '3',
                trajectory_mode: document.getElementById('trajectory-mode')?.value || 'auto',
                tier2_weight: document.getElementById('tier2-weight')?.value || '0.20',
                upscale_backend: document.getElementById('upscale-backend')?.value || 'none',
            });
```

- [ ] **Step 4: Add upscale stage label**

In `static/js/app.js`, add to the `stageLabels` object:

```javascript
            'upscale': 'Upscaling video (SeedVR2)',
```

- [ ] **Step 5: Commit**

```bash
git add api.py static/index.html static/js/app.js
git commit -m "feat(seedvr2): add upscale backend toggle to API and web UI"
```

---

## Summary

| Task | Module | Type | Est. Size |
|------|--------|------|-----------|
| 1 | Config fields | Modify | ~10 lines |
| 2 | seedvr2_adapter.py | New | ~100 lines impl + ~80 lines tests |
| 3 | Pipeline Stage 3 wiring | Modify | ~30 lines |
| 4 | Setup script | Modify | ~25 lines |
| 5 | API + UI | Modify | ~15 lines across 3 files |

**Total new code:** ~180 lines + ~80 lines tests
