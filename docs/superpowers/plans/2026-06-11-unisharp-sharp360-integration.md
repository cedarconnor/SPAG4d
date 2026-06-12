# UniSHARP `sharp360` Backend Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `unisharp` backend to SPAG4d's `sharp360` generator that runs Insta360 UniSHARP's native-ERP panorama inference in its own isolated environment (subprocess) and emits a 3DGS PLY into SPAG4d's output path.

**Architecture:** A subprocess adapter calls UniSHARP's `scripts/infer_unisharp.py` (with `--save-ply --camera panorama`, `cwd=repo`) inside its own conda env. The resulting `gaussians.ply` is located by glob, then either copied verbatim (`copy` mode) or rewritten as a clean INRIA PLY (`convert` mode). `convert_sharp360` gains a `backend` parameter that dispatches `"unisharp"` to the new wrapper while leaving the existing per-face SHARP path (`backend="sharp"`) untouched. `"hybrid"` raises `NotImplementedError`.

**Tech Stack:** Python 3.10+ (SPAG4d side), `subprocess`, `plyfile` (already a dependency via `ply_writer.py`), Click CLI, pytest. UniSHARP runs out-of-process under Python 3.12 / torch 2.8 — never imported into SPAG4d.

---

## Scope & Source

This plan implements **Milestones M1, M1.5, and M3** from `docs/SPAG4d_UniSHARP_Integration_Design_v2.md`. Those three form one shippable unit: a working UniSHARP backend with copy and convert PLY modes, plus a manual viewer reality-check gate. Scale alignment (M4/M5) and hybrid merging (M6) are deferred to separate follow-up plans (see "Deferred Work" at the end) because they require live GPU research iteration, not just code.

**Everything in M1/M3 is unit-testable without a GPU or a UniSHARP install** by mocking `subprocess.run` and using synthetic PLY fixtures. The only step that needs the real environment is the M1.5 smoke/viewer check, which is a documented manual runbook (Task 10), explicitly gated.

### Corrections applied vs. the design doc

The doc's code snippets are mostly accurate but assume an argparse-style CLI and an abbreviated `core.py` call. The real codebase differs:

1. **CLI is Click**, not argparse. New options are `@click.option` decorators on the `convert` command in `spag4d/cli.py`, threaded through `convert()` → `run_single` → `SPAG4D.convert()`.
2. **`core.py` already passes** `include_caps`, `cap_fov_degrees`, `seam_latitude_degrees` to `convert_sharp360`. The new backend args are **added** to that existing call (the doc's §10.2 snippet drops those args — do not follow it literally).
3. **`convert_sharp360` returns** `{"num_gaussians", "num_faces", "output_path"}` only (no `processing_time`); `core.py` reads `processing_time` via `.get(..., 0.0)`. The unisharp path returns a superset, which is compatible.
4. **`color_space` PLY encoding is unverified.** PLY has no native string property, so how UniSHARP stores `sRGB`/`linearRGB` is uncertain. The reader (`_read_color_space`) tries PLY comments first and returns `None` on failure; `convert` mode **preserves** `f_dc` colors by default (no color-space conversion), so M3 correctness does not depend on reading it. Confirm the real encoding during M1.5 (Task 10) and adjust then.

---

## File Structure

**New files:**

| File | Responsibility |
|------|----------------|
| `spag4d/unisharp_adapter.py` | Build & run the `infer_unisharp.py` subprocess (`cwd=repo`, `--save-ply`, `--camera panorama`); locate the output PLY by glob; raise readable errors. |
| `spag4d/unisharp_format.py` | Inspect PLY fields, count vertices, copy raw PLY (`copy` mode), rewrite clean INRIA PLY (`convert` mode). |
| `spag4d/unisharp360.py` | High-level wrapper: resolve env fallbacks, validate inputs, manage work dir, call adapter, apply format mode, save debug artifacts, return stats dict. |
| `tests/_unisharp_fixtures.py` | Shared helper that writes a synthetic UniSHARP-style PLY (core vertex fields + supplement elements + `color_space` comment). |
| `tests/test_unisharp_format.py` | Unit tests for inspect/count/copy/convert. |
| `tests/test_unisharp_adapter.py` | Unit tests for subprocess command construction & error paths (mocked `subprocess.run`). |
| `tests/test_unisharp360.py` | Unit tests for validation, env fallbacks, and the full wrapper flow (mocked adapter). |
| `tests/test_unisharp_dispatch.py` | Unit tests for `convert_sharp360` backend dispatch + `SPAG4D.convert` routing (mocked). |

**Modified files:**

| File | Change |
|------|--------|
| `spag4d/sharp360.py` | Add `backend` + `unisharp_*` params to `convert_sharp360`; dispatch `"unisharp"`/`"hybrid"` before the existing SHARP body. |
| `spag4d/core.py` | Add `sharp_backend` + `unisharp_*` params to `SPAG4D.convert`; extend the `sharp360` dispatch to also handle `unisharp360` and pass the new args. |
| `spag4d/cli.py` | Add `unisharp360` to `--generator` choices; add `--sharp-backend` + `--unisharp-*` Click options; thread them through `run_single`. |

**Conventions to match the existing codebase:**
- Module-level `LOGGER = logging.getLogger(__name__)` (see `sharp360.py`).
- `from __future__ import annotations` at the top of new modules.
- Tests use pytest with class-based grouping (see `tests/test_sharp360.py`).
- Run tests with the venv Python: `.venv\Scripts\python.exe -m pytest ...` (Windows).
- Lazy imports inside functions for heavy/optional deps (the existing `convert_sharp360` imports `sharp` lazily — our dispatch must branch to `unisharp` **before** that import so the unisharp path never imports `sharp`).

---

## Task 1: Synthetic PLY fixture + `unisharp_format.py` inspect/count

**Files:**
- Create: `tests/_unisharp_fixtures.py`
- Create: `spag4d/unisharp_format.py`
- Test: `tests/test_unisharp_format.py`

- [ ] **Step 1: Write the shared fixture helper**

Create `tests/_unisharp_fixtures.py`:

```python
"""Helpers that synthesize a UniSHARP-style 3DGS PLY for tests.

The real UniSHARP writer emits a standard INRIA vertex element plus extra
"supplement" elements (extrinsic / intrinsic / image_size) and a color_space
marker. PLY has no native string property, so we encode color_space as a PLY
comment here; the real encoding is verified during the M1.5 viewer check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

CORE_VERTEX_FIELDS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
    + ["opacity"]
)


def write_fake_unisharp_ply(
    path: str | Path,
    n_vertices: int = 25,
    with_supplements: bool = True,
    color_space: str | None = "sRGB",
) -> Path:
    """Write a UniSHARP-shaped PLY and return its path."""
    path = Path(path)
    rng = np.random.RandomState(0)  # deterministic; no Math.random/Date needed

    dtype = [(name, "f4") for name in CORE_VERTEX_FIELDS]
    vtx = np.empty(n_vertices, dtype=dtype)
    for name in CORE_VERTEX_FIELDS:
        vtx[name] = rng.rand(n_vertices).astype("f4")
    # Make quaternions look normalized-ish so the file is plausible.
    vtx["rot_0"] = 1.0

    elements = [PlyElement.describe(vtx, "vertex")]

    if with_supplements:
        extr = np.empty(4, dtype=[(f"m{i}", "f4") for i in range(4)])
        for i in range(4):
            extr[f"m{i}"] = np.eye(4, dtype="f4")[:, i]
        intr = np.empty(3, dtype=[(f"m{i}", "f4") for i in range(3)])
        for i in range(3):
            intr[f"m{i}"] = np.eye(3, dtype="f4")[:, i]
        size = np.array([(1536, 768)], dtype=[("width", "i4"), ("height", "i4")])
        elements.append(PlyElement.describe(extr, "extrinsic"))
        elements.append(PlyElement.describe(intr, "intrinsic"))
        elements.append(PlyElement.describe(size, "image_size"))

    comments = []
    if color_space is not None:
        comments.append(f"color_space {color_space}")

    PlyData(elements, text=False, comments=comments).write(str(path))
    return path
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_unisharp_format.py`:

```python
"""Tests for spag4d.unisharp_format — PLY inspect/count/copy/convert."""
from pathlib import Path

import pytest

from tests._unisharp_fixtures import CORE_VERTEX_FIELDS, write_fake_unisharp_ply
from spag4d.unisharp_format import (
    count_ply_vertices,
    inspect_ply_fields,
)


class TestInspect:
    def test_detects_core_fields(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", n_vertices=10)
        info = inspect_ply_fields(str(ply))
        assert info["has_core_fields"] is True
        for f in CORE_VERTEX_FIELDS:
            assert f in info["vertex_fields"]

    def test_detects_supplement_elements(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", with_supplements=True)
        info = inspect_ply_fields(str(ply))
        assert set(info["supplement_elements"]) == {"extrinsic", "intrinsic", "image_size"}

    def test_no_supplements_when_absent(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", with_supplements=False)
        info = inspect_ply_fields(str(ply))
        assert info["supplement_elements"] == []

    def test_reads_color_space_comment(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", color_space="linearRGB")
        info = inspect_ply_fields(str(ply))
        assert info["color_space"] == "linearRGB"

    def test_color_space_none_when_absent(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", color_space=None)
        info = inspect_ply_fields(str(ply))
        assert info["color_space"] is None


class TestCount:
    def test_counts_vertices(self, tmp_path):
        ply = write_fake_unisharp_ply(tmp_path / "g.ply", n_vertices=42)
        assert count_ply_vertices(str(ply)) == 42
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spag4d.unisharp_format'`

- [ ] **Step 4: Write minimal implementation**

Create `spag4d/unisharp_format.py`:

```python
"""UniSHARP PLY format handling: inspect, count, copy, convert."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)

CORE_VERTEX_FIELDS = (
    ["x", "y", "z"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
    + ["opacity"]
)


def _read_color_space(ply) -> str | None:
    """Best-effort color_space read. Tries PLY comments first.

    The real UniSHARP encoding is unverified (PLY has no string property type),
    so this is tolerant: it returns None rather than raising when absent.
    """
    for c in getattr(ply, "comments", []) or []:
        parts = c.strip().split()
        if len(parts) == 2 and parts[0] == "color_space":
            return parts[1]
    return None


def inspect_ply_fields(ply_path: str) -> dict:
    """Report element names, vertex field names, and supplement elements."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    elements = {el.name: [p.name for p in el.properties] for el in ply.elements}
    vertex_fields = elements.get("vertex", [])
    supplements = [name for name in elements if name != "vertex"]
    return {
        "elements": elements,
        "vertex_fields": vertex_fields,
        "supplement_elements": supplements,
        "has_core_fields": all(f in vertex_fields for f in CORE_VERTEX_FIELDS),
        "color_space": _read_color_space(ply),
    }


def count_ply_vertices(ply_path: str) -> int:
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    for el in ply.elements:
        if el.name == "vertex":
            return int(el.count)
    return 0
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_format.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add spag4d/unisharp_format.py tests/_unisharp_fixtures.py tests/test_unisharp_format.py
git commit -m "feat(unisharp): PLY inspect/count + synthetic fixture"
```

---

## Task 2: `unisharp_format.py` — copy mode

**Files:**
- Modify: `spag4d/unisharp_format.py`
- Test: `tests/test_unisharp_format.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unisharp_format.py`:

```python
from spag4d.unisharp_format import copy_unisharp_ply_to_output


class TestCopy:
    def test_copy_preserves_vertices_and_supplements(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=17,
                                      with_supplements=True)
        dst = tmp_path / "out.ply"
        stats = copy_unisharp_ply_to_output(str(src), str(dst))
        assert dst.exists()
        assert stats["num_gaussians"] == 17
        assert set(stats["ply_info"]["supplement_elements"]) == {
            "extrinsic", "intrinsic", "image_size"
        }

    def test_copy_is_byte_identical(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=8)
        dst = tmp_path / "out.ply"
        copy_unisharp_ply_to_output(str(src), str(dst))
        assert dst.read_bytes() == src.read_bytes()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_format.py::TestCopy -v`
Expected: FAIL — `ImportError: cannot import name 'copy_unisharp_ply_to_output'`

- [ ] **Step 3: Write minimal implementation**

Append to `spag4d/unisharp_format.py`:

```python
def copy_unisharp_ply_to_output(src_path: str, dst_path: str) -> dict:
    """Copy raw UniSHARP PLY verbatim and count vertices.

    Keeps supplement elements intact. Works in viewers that tolerate extra PLY
    elements (SuperSplat, GaussianSplats3D). Use convert mode for strict loaders.
    """
    shutil.copy2(src_path, dst_path)
    n = count_ply_vertices(dst_path)
    info = inspect_ply_fields(dst_path)
    if info["supplement_elements"]:
        LOGGER.info(
            "UniSHARP PLY carries supplement elements %s; "
            "strict loaders may need format_mode='convert'.",
            info["supplement_elements"],
        )
    return {"num_gaussians": n, "ply_info": info}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_format.py::TestCopy -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add spag4d/unisharp_format.py tests/test_unisharp_format.py
git commit -m "feat(unisharp): copy-mode PLY passthrough"
```

---

## Task 3: `unisharp_format.py` — convert mode

**Files:**
- Modify: `spag4d/unisharp_format.py`
- Test: `tests/test_unisharp_format.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unisharp_format.py`:

```python
from spag4d.unisharp_format import convert_unisharp_ply_to_spag


class TestConvert:
    def test_convert_drops_supplements_keeps_core(self, tmp_path):
        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=20,
                                      with_supplements=True)
        dst = tmp_path / "out.ply"
        stats = convert_unisharp_ply_to_spag(str(src), str(dst))
        assert stats["num_gaussians"] == 20
        assert stats["converted"] is True

        info = inspect_ply_fields(str(dst))
        assert info["supplement_elements"] == []
        assert info["has_core_fields"] is True

    def test_convert_preserves_vertex_values(self, tmp_path):
        from plyfile import PlyData
        import numpy as np

        src = write_fake_unisharp_ply(tmp_path / "src.ply", n_vertices=12)
        dst = tmp_path / "out.ply"
        convert_unisharp_ply_to_spag(str(src), str(dst))

        a = PlyData.read(str(src))["vertex"]
        b = PlyData.read(str(dst))["vertex"]
        for f in CORE_VERTEX_FIELDS:
            np.testing.assert_allclose(np.asarray(a[f]), np.asarray(b[f]), rtol=1e-6)

    def test_convert_raises_on_missing_core_field(self, tmp_path):
        from plyfile import PlyData, PlyElement
        import numpy as np

        # A PLY missing opacity should raise KeyError.
        bad_fields = [f for f in CORE_VERTEX_FIELDS if f != "opacity"]
        arr = np.zeros(5, dtype=[(n, "f4") for n in bad_fields])
        bad = tmp_path / "bad.ply"
        PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(bad))

        with pytest.raises(KeyError, match="opacity"):
            convert_unisharp_ply_to_spag(str(bad), str(tmp_path / "out.ply"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_format.py::TestConvert -v`
Expected: FAIL — `ImportError: cannot import name 'convert_unisharp_ply_to_spag'`

- [ ] **Step 3: Write minimal implementation**

Append to `spag4d/unisharp_format.py`:

```python
def convert_unisharp_ply_to_spag(src_path: str, dst_path: str) -> dict:
    """Rewrite UniSHARP PLY as a clean INRIA 3DGS PLY.

    Drops supplement elements, keeps exactly CORE_VERTEX_FIELDS in order.
    Scales stay log, opacity stays logit, quats stay wxyz. Colors (f_dc) are
    preserved as-is by default; color_space conversion is deferred until the
    real encoding is confirmed (M1.5).
    """
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(src_path)
    vtx = next(el for el in ply.elements if el.name == "vertex")
    present = {p.name for p in vtx.properties}

    missing = [f for f in CORE_VERTEX_FIELDS if f not in present]
    if missing:
        raise KeyError(f"UniSHARP PLY missing core fields: {missing}")

    dtype = [(name, "f4") for name in CORE_VERTEX_FIELDS]
    arr = np.empty(int(vtx.count), dtype=dtype)
    for name in CORE_VERTEX_FIELDS:
        arr[name] = np.asarray(vtx[name]).astype("f4")

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(dst_path)
    return {"num_gaussians": int(len(arr)), "converted": True}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_format.py -v`
Expected: PASS (all format tests)

- [ ] **Step 5: Commit**

```bash
git add spag4d/unisharp_format.py tests/test_unisharp_format.py
git commit -m "feat(unisharp): convert-mode clean INRIA PLY rewrite"
```

---

## Task 4: `unisharp_adapter.py` — subprocess command construction

**Files:**
- Create: `spag4d/unisharp_adapter.py`
- Test: `tests/test_unisharp_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_unisharp_adapter.py`:

```python
"""Tests for spag4d.unisharp_adapter — subprocess command construction."""
import subprocess
import types
from pathlib import Path

import pytest

from tests._unisharp_fixtures import write_fake_unisharp_ply
from spag4d import unisharp_adapter
from spag4d.unisharp_adapter import run_unisharp_inference


def _make_repo(tmp_path):
    """Create a fake UniSHARP repo with the inference script present."""
    repo = tmp_path / "UniSHARP"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer_unisharp.py").write_text("# stub\n")
    return repo


def _fake_run_factory(captured, out_dir, returncode=0, write_ply=True):
    """Return a fake subprocess.run that records args and writes a PLY."""
    def _fake_run(cmd, cwd=None, capture_output=False, text=False, timeout=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        if write_ply and returncode == 0:
            sample = Path(out_dir) / "panos_pano"
            sample.mkdir(parents=True, exist_ok=True)
            write_fake_unisharp_ply(sample / "gaussians.ply")
            (sample / "metadata.json").write_text("{}")
        return types.SimpleNamespace(
            returncode=returncode, stdout="ok-stdout", stderr="err-stderr"
        )
    return _fake_run


class TestCommandConstruction:
    def test_includes_save_ply_and_camera_panorama(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir))

        result = run_unisharp_inference(
            image_path=str(tmp_path / "pano.jpg"),
            out_dir=str(out_dir),
            repo_dir=str(repo),
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            python_exe="python",
        )
        cmd = captured["cmd"]
        assert "--save-ply" in cmd
        assert "--camera" in cmd and cmd[cmd.index("--camera") + 1] == "panorama"
        assert "--checkpoint" in cmd
        assert result["ply_path"].endswith("gaussians.ply")
        assert result["metadata_path"].endswith("metadata.json")

    def test_cwd_is_repo_root(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir))
        run_unisharp_inference(
            image_path=str(tmp_path / "pano.jpg"), out_dir=str(out_dir),
            repo_dir=str(repo), checkpoint_path=str(tmp_path / "ckpt.pt"),
        )
        assert captured["cwd"] == str(repo)


class TestErrorPaths:
    def test_missing_script_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="inference script not found"):
            run_unisharp_inference(
                image_path="x.jpg", out_dir=str(tmp_path / "out"),
                repo_dir=str(tmp_path / "nope"), checkpoint_path="c.pt",
            )

    def test_nonzero_returncode_raises_with_logs(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir, returncode=1,
                                              write_ply=False))
        with pytest.raises(RuntimeError, match="UniSHARP inference failed"):
            run_unisharp_inference(
                image_path="x.jpg", out_dir=str(out_dir),
                repo_dir=str(repo), checkpoint_path="c.pt",
            )

    def test_no_ply_raises(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        out_dir = tmp_path / "out"
        captured = {}
        monkeypatch.setattr(subprocess, "run",
                            _fake_run_factory(captured, out_dir, write_ply=False))
        with pytest.raises(FileNotFoundError, match="no gaussians.ply"):
            run_unisharp_inference(
                image_path="x.jpg", out_dir=str(out_dir),
                repo_dir=str(repo), checkpoint_path="c.pt",
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spag4d.unisharp_adapter'`

- [ ] **Step 3: Write minimal implementation**

Create `spag4d/unisharp_adapter.py`:

```python
"""Subprocess adapter around UniSHARP's scripts/infer_unisharp.py."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def run_unisharp_inference(
    image_path: str,
    out_dir: str,
    repo_dir: str,
    checkpoint_path: str,
    python_exe: Optional[str] = None,
    camera: str = "panorama",
    device: str = "cuda:0",
    save_ply: bool = True,
    max_images: int = 1,
    extra_args: Optional[list] = None,
    timeout_s: Optional[float] = None,
) -> dict:
    """Invoke UniSHARP's infer_unisharp.py as a subprocess.

    Returns dict with returncode, stdout, stderr, out_dir, ply_path,
    metadata_path. Raises FileNotFoundError if the script or output PLY is
    missing, and RuntimeError on a non-zero exit.
    """
    repo = Path(repo_dir)
    script = repo / "scripts" / "infer_unisharp.py"
    if not script.exists():
        raise FileNotFoundError(f"UniSHARP inference script not found: {script}")

    python = python_exe or sys.executable

    cmd = [
        python, str(script),
        "--checkpoint", str(checkpoint_path),
        "--image", str(image_path),
        "--out-dir", str(out_dir),
        "--camera", camera,
        "--device", device,
        "--max-images", str(max_images),
    ]
    if save_ply:
        cmd.append("--save-ply")  # REQUIRED or no PLY is written
    if extra_args:
        cmd.extend(extra_args)

    LOGGER.info("Running UniSHARP: %s", " ".join(cmd))

    # cwd MUST be the repo root: the script does sys.path.insert(REPO_ROOT),
    # writes checkpoints/torchhub there, and expects ./UniK3D as a sibling.
    proc = subprocess.run(
        cmd, cwd=str(repo),
        capture_output=True, text=True, timeout=timeout_s,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "UniSHARP inference failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )

    out = Path(out_dir)
    ply_candidates = sorted(out.glob("**/gaussians.ply"))
    meta_candidates = sorted(out.glob("**/metadata.json"))
    if not ply_candidates:
        raise FileNotFoundError(
            f"UniSHARP completed but no gaussians.ply under {out}. "
            "Confirm --save-ply was passed and inference produced a sample dir."
        )
    if len(ply_candidates) > 1:
        LOGGER.warning("Multiple PLYs found; using first: %s", ply_candidates[0])

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_dir": str(out),
        "ply_path": str(ply_candidates[0]),
        "metadata_path": str(meta_candidates[0]) if meta_candidates else None,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_adapter.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add spag4d/unisharp_adapter.py tests/test_unisharp_adapter.py
git commit -m "feat(unisharp): subprocess adapter for infer_unisharp.py"
```

---

## Task 5: `unisharp360.py` — input validation & env fallbacks

**Files:**
- Create: `spag4d/unisharp360.py`
- Test: `tests/test_unisharp360.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_unisharp360.py`:

```python
"""Tests for spag4d.unisharp360 — wrapper validation and flow."""
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests._unisharp_fixtures import write_fake_unisharp_ply
from spag4d.unisharp360 import convert_unisharp360


def _erp_image(path, w=512, h=256):
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)
    return str(path)


def _make_repo(tmp_path):
    repo = tmp_path / "UniSHARP"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer_unisharp.py").write_text("# stub\n")
    return repo


CPU = torch.device("cpu")


class TestValidation:
    def test_missing_repo_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPAG4D_UNISHARP_REPO", raising=False)
        img = _erp_image(tmp_path / "pano.jpg")
        with pytest.raises(ValueError, match="requires --unisharp-repo"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=None, checkpoint_path="c.pt",
            )

    def test_missing_repo_dir_raises(self, tmp_path):
        img = _erp_image(tmp_path / "pano.jpg")
        with pytest.raises(FileNotFoundError, match="repo not found"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=str(tmp_path / "nope"),
                checkpoint_path="c.pt",
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        repo = _make_repo(tmp_path)
        img = _erp_image(tmp_path / "pano.jpg")
        with pytest.raises(ValueError, match="requires --unisharp-checkpoint"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=str(repo), checkpoint_path=None,
            )

    def test_wrong_aspect_raises(self, tmp_path):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"
        ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "sq.jpg", w=256, h=256)  # 1:1, not 2:1
        with pytest.raises(ValueError, match="2:1 ERP"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            )

    def test_env_fallbacks_used(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"
        ckpt.write_bytes(b"x")
        monkeypatch.setenv("SPAG4D_UNISHARP_REPO", str(repo))
        monkeypatch.setenv("SPAG4D_UNISHARP_CHECKPOINT", str(ckpt))
        img = _erp_image(tmp_path / "sq.jpg", w=256, h=256)  # 1:1 -> fails AFTER env resolve
        # If env fallbacks did NOT resolve, we'd get "requires --unisharp-repo"
        # instead of the aspect error. Asserting the aspect error proves
        # env resolution happened.
        with pytest.raises(ValueError, match="2:1 ERP"):
            convert_unisharp360(
                input_path=img, output_path=str(tmp_path / "o.ply"),
                device=CPU, unisharp_repo=None, checkpoint_path=None,
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp360.py::TestValidation -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spag4d.unisharp360'`

- [ ] **Step 3: Write minimal implementation**

Create `spag4d/unisharp360.py`:

```python
"""High-level SPAG4d-facing wrapper for the UniSHARP backend."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)


def convert_unisharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    unisharp_repo: Optional[str],
    unisharp_python: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    scale_align: str = "global",
    format_mode: str = "copy",
    save_debug: bool = False,
    raw_output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    t0 = time.time()

    # ---- resolve env fallbacks -------------------------------------------
    unisharp_repo = unisharp_repo or os.environ.get("SPAG4D_UNISHARP_REPO")
    unisharp_python = unisharp_python or os.environ.get("SPAG4D_UNISHARP_PYTHON")
    checkpoint_path = checkpoint_path or os.environ.get("SPAG4D_UNISHARP_CHECKPOINT")

    # ---- validation -------------------------------------------------------
    if not unisharp_repo:
        raise ValueError(
            "UniSHARP backend requires --unisharp-repo (or SPAG4D_UNISHARP_REPO) "
            "pointing to a local clone of Insta360-Research-Team/UniSHARP."
        )
    repo = Path(unisharp_repo)
    if not repo.exists():
        raise FileNotFoundError(f"UniSHARP repo not found: {repo}")
    if not checkpoint_path:
        raise ValueError(
            "UniSHARP backend requires --unisharp-checkpoint "
            "(or SPAG4D_UNISHARP_CHECKPOINT)."
        )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"UniSHARP checkpoint not found: {checkpoint_path}")

    with Image.open(input_path) as im:
        w, h = im.size
    if abs((w / h) - 2.0) > 0.05:
        raise ValueError(
            f"UniSHARP panorama mode expects a 2:1 ERP image. Got {w}x{h}. "
            "Use --sharp-backend sharp for non-ERP inputs."
        )

    # ---- working directory ------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_output_dir:
        work_dir = Path(raw_output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="spag4d_unisharp_")
        work_dir = Path(tmp_ctx.name)

    try:
        if progress_callback:
            progress_callback("unisharp_inference", 0, 1)

        from .unisharp_adapter import run_unisharp_inference
        run = run_unisharp_inference(
            image_path=input_path,
            out_dir=str(work_dir / "unisharp_out"),
            repo_dir=str(repo),
            checkpoint_path=str(checkpoint_path),
            python_exe=unisharp_python,
            camera="panorama",
            device=("cuda:0" if device.type == "cuda" else "cpu"),
            save_ply=True,
            max_images=1,
        )
        raw_ply = Path(run["ply_path"])

        if progress_callback:
            progress_callback("unisharp_inference", 1, 1)

        # ---- format handling ---------------------------------------------
        from .unisharp_format import (
            copy_unisharp_ply_to_output,
            convert_unisharp_ply_to_spag,
        )
        if format_mode == "copy":
            stats = copy_unisharp_ply_to_output(str(raw_ply), str(out_path))
        elif format_mode == "convert":
            stats = convert_unisharp_ply_to_spag(str(raw_ply), str(out_path))
        else:
            raise ValueError(f"Unknown format_mode {format_mode!r}")

        # ---- optional scale alignment ------------------------------------
        # scale_align in {"none","global","da360_grid"}. M1 ships "none"/"global"
        # as a no-op-safe passthrough; real alignment lands in a follow-up plan.

        # ---- debug artifacts ---------------------------------------------
        if save_debug:
            dbg = out_path.parent / (out_path.stem + "_unisharp_debug")
            dbg.mkdir(parents=True, exist_ok=True)
            for art in ("forward.gif", "rotate.gif", "metadata.json"):
                src = raw_ply.parent / art
                if src.exists():
                    shutil.copy2(src, dbg / art)
            shutil.copy2(raw_ply, dbg / "raw_gaussians.ply")

        return {
            "num_gaussians": stats["num_gaussians"],
            "num_faces": 0,  # native ERP: no faces
            "output_path": str(out_path),
            "processing_time": time.time() - t0,
            "backend": "unisharp",
        }
    finally:
        if tmp_ctx is not None and not save_debug:
            tmp_ctx.cleanup()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp360.py::TestValidation -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add spag4d/unisharp360.py tests/test_unisharp360.py
git commit -m "feat(unisharp): wrapper validation + env fallbacks"
```

---

## Task 6: `unisharp360.py` — full flow with mocked adapter

**Files:**
- Modify: `tests/test_unisharp360.py` (no production change — verifies Task 5 code end-to-end)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unisharp360.py`:

```python
class TestFlow:
    def _patch_adapter(self, monkeypatch, tmp_path):
        """Patch run_unisharp_inference to emit a fixture PLY + artifacts."""
        import spag4d.unisharp360 as mod

        def _fake_run(**kwargs):
            out = Path(kwargs["out_dir"]) / "panos_pano"
            out.mkdir(parents=True, exist_ok=True)
            ply = out / "gaussians.ply"
            write_fake_unisharp_ply(ply, n_vertices=33, with_supplements=True)
            (out / "metadata.json").write_text("{}")
            (out / "forward.gif").write_bytes(b"GIF")
            (out / "rotate.gif").write_bytes(b"GIF")
            return {"returncode": 0, "stdout": "", "stderr": "",
                    "out_dir": kwargs["out_dir"], "ply_path": str(ply),
                    "metadata_path": str(out / "metadata.json")}

        monkeypatch.setattr(mod, "run_unisharp_inference", _fake_run,
                            raising=False)
        # The function imports the symbol lazily; patch the source module too.
        import spag4d.unisharp_adapter as adapter_mod
        monkeypatch.setattr(adapter_mod, "run_unisharp_inference", _fake_run)

    def test_copy_mode_full_flow(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "pano.jpg")
        self._patch_adapter(monkeypatch, tmp_path)

        out = tmp_path / "out.ply"
        stats = convert_unisharp360(
            input_path=img, output_path=str(out), device=CPU,
            unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            format_mode="copy",
        )
        assert out.exists()
        assert stats["num_gaussians"] == 33
        assert stats["num_faces"] == 0
        assert stats["backend"] == "unisharp"

    def test_save_debug_copies_artifacts(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "pano.jpg")
        self._patch_adapter(monkeypatch, tmp_path)

        out = tmp_path / "out.ply"
        convert_unisharp360(
            input_path=img, output_path=str(out), device=CPU,
            unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            save_debug=True,
        )
        dbg = tmp_path / "out_unisharp_debug"
        assert (dbg / "raw_gaussians.ply").exists()
        assert (dbg / "metadata.json").exists()
        assert (dbg / "forward.gif").exists()

    def test_convert_mode_strips_supplements(self, tmp_path, monkeypatch):
        from spag4d.unisharp_format import inspect_ply_fields
        repo = _make_repo(tmp_path)
        ckpt = tmp_path / "c.pt"; ckpt.write_bytes(b"x")
        img = _erp_image(tmp_path / "pano.jpg")
        self._patch_adapter(monkeypatch, tmp_path)

        out = tmp_path / "out.ply"
        convert_unisharp360(
            input_path=img, output_path=str(out), device=CPU,
            unisharp_repo=str(repo), checkpoint_path=str(ckpt),
            format_mode="convert",
        )
        assert inspect_ply_fields(str(out))["supplement_elements"] == []
```

Note: `convert_unisharp360` does `from .unisharp_adapter import run_unisharp_inference` **inside** the function, so the binding resolves from `spag4d.unisharp_adapter` at call time. Patching `spag4d.unisharp_adapter.run_unisharp_inference` (the second `setattr` above) is the one that takes effect.

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp360.py::TestFlow -v`
Expected: PASS (3 tests) — Task 5's implementation already supports this flow. If any fail, fix `unisharp360.py` until green (this task is the end-to-end proof of Task 5).

- [ ] **Step 3: Run the full unisharp suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp360.py tests/test_unisharp_format.py tests/test_unisharp_adapter.py -v`
Expected: PASS (all)

- [ ] **Step 4: Commit**

```bash
git add tests/test_unisharp360.py
git commit -m "test(unisharp): end-to-end wrapper flow with mocked adapter"
```

---

## Task 7: `sharp360.py` — backend dispatch

**Files:**
- Modify: `spag4d/sharp360.py:906-918` (signature) and the line immediately after the docstring (insert dispatch)
- Test: `tests/test_unisharp_dispatch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_unisharp_dispatch.py`:

```python
"""Tests for convert_sharp360 backend dispatch."""
import types

import pytest
import torch

import spag4d.sharp360 as sharp360
from spag4d.sharp360 import convert_sharp360

CPU = torch.device("cpu")


class TestBackendDispatch:
    def test_unisharp_backend_dispatches(self, monkeypatch):
        called = {}

        def _fake_convert_unisharp360(**kwargs):
            called.update(kwargs)
            return {"num_gaussians": 7, "num_faces": 0,
                    "output_path": kwargs["output_path"],
                    "processing_time": 0.0, "backend": "unisharp"}

        # convert_sharp360 imports convert_unisharp360 lazily from the module.
        import spag4d.unisharp360 as uni
        monkeypatch.setattr(uni, "convert_unisharp360", _fake_convert_unisharp360)

        result = convert_sharp360(
            input_path="pano.jpg", output_path="out.ply", device=CPU,
            backend="unisharp", unisharp_repo="/repo",
            unisharp_checkpoint="/ckpt.pt",
        )
        assert result["num_gaussians"] == 7
        assert called["unisharp_repo"] == "/repo"
        assert called["checkpoint_path"] == "/ckpt.pt"

    def test_hybrid_backend_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="hybrid"):
            convert_sharp360(
                input_path="pano.jpg", output_path="out.ply", device=CPU,
                backend="hybrid",
            )

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown sharp360 backend"):
            convert_sharp360(
                input_path="pano.jpg", output_path="out.ply", device=CPU,
                backend="bogus",
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_dispatch.py::TestBackendDispatch -v`
Expected: FAIL — `TypeError: convert_sharp360() got an unexpected keyword argument 'backend'`

- [ ] **Step 3: Extend the signature**

In `spag4d/sharp360.py`, change the `convert_sharp360` signature (currently lines 906-918) to add the new parameters after `seam_latitude_degrees`:

```python
def convert_sharp360(
    input_path: str,
    output_path: str,
    device: torch.device,
    side_count: int = 6,
    overlap_degrees: float = 10.0,
    seedvr2_upscale: bool = False,
    seedvr2_config: Optional["SeedVR2Config"] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    include_caps: bool = True,
    cap_fov_degrees: float = 125.0,
    seam_latitude_degrees: float = 30.0,
    # --- new: backend selection ---
    backend: str = "sharp",
    unisharp_repo: Optional[str] = None,
    unisharp_python: Optional[str] = None,
    unisharp_checkpoint: Optional[str] = None,
    unisharp_scale_align: str = "global",
    unisharp_format_mode: str = "copy",
    unisharp_save_debug: bool = False,
    unisharp_raw_output_dir: Optional[str] = None,
) -> dict:
```

- [ ] **Step 4: Insert the dispatch block**

Immediately after the docstring closes (after the `"""..."""` block ending at line 933) and **before** the existing `from sharp.cli.predict import predict_image` line, insert:

```python
    backend = str(backend).lower().strip()
    if backend not in {"sharp", "unisharp", "hybrid"}:
        raise ValueError(
            f"Unknown sharp360 backend {backend!r}. "
            "Expected 'sharp', 'unisharp', or 'hybrid'."
        )

    if backend == "unisharp":
        from .unisharp360 import convert_unisharp360
        return convert_unisharp360(
            input_path=input_path,
            output_path=output_path,
            device=device,
            unisharp_repo=unisharp_repo,
            unisharp_python=unisharp_python,
            checkpoint_path=unisharp_checkpoint,
            scale_align=unisharp_scale_align,
            format_mode=unisharp_format_mode,
            save_debug=unisharp_save_debug,
            raw_output_dir=unisharp_raw_output_dir,
            progress_callback=progress_callback,
        )

    if backend == "hybrid":
        raise NotImplementedError(
            "sharp360 backend='hybrid' is planned but not implemented. "
            "Use 'sharp' or 'unisharp'."
        )

    # backend == "sharp": existing per-face pipeline (below), untouched.
```

This places the dispatch ahead of the `from sharp.cli.predict import predict_image` import, so the `unisharp` and `hybrid` branches never import `sharp`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_dispatch.py::TestBackendDispatch -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Verify the existing SHARP path is untouched**

Run: `.venv\Scripts\python.exe -m pytest tests/test_sharp360.py -v`
Expected: PASS (all existing tests still green — the `sharp` default falls through unchanged)

- [ ] **Step 7: Commit**

```bash
git add spag4d/sharp360.py tests/test_unisharp_dispatch.py
git commit -m "feat(sharp360): add backend dispatch (sharp/unisharp/hybrid)"
```

---

## Task 8: `core.py` — convert() params + unisharp360 routing

**Files:**
- Modify: `spag4d/core.py:82-108` (convert signature) and `spag4d/core.py:156-181` (dispatch)
- Test: `tests/test_unisharp_dispatch.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unisharp_dispatch.py`:

```python
class TestCoreRouting:
    def test_generator_unisharp360_maps_to_backend(self, tmp_path, monkeypatch):
        """SPAG4D.convert(generator='unisharp360') routes to convert_sharp360
        with backend='unisharp'."""
        import numpy as np
        from PIL import Image
        import spag4d.core as core_mod

        # 2:1 ERP so the aspect gate passes.
        img = tmp_path / "pano.jpg"
        Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8)).save(img)
        out = tmp_path / "out.ply"

        captured = {}

        def _fake_convert_sharp360(**kwargs):
            captured.update(kwargs)
            out_p = kwargs["output_path"]
            # Produce a real file so the .stat() in core.py succeeds.
            from tests._unisharp_fixtures import write_fake_unisharp_ply
            write_fake_unisharp_ply(out_p, n_vertices=5)
            return {"num_gaussians": 5, "num_faces": 0, "output_path": out_p}

        monkeypatch.setattr(core_mod, "convert_sharp360",
                            _fake_convert_sharp360, raising=False)
        import spag4d.sharp360 as s
        monkeypatch.setattr(s, "convert_sharp360", _fake_convert_sharp360)

        conv = core_mod.SPAG4D(device="cpu", generator="unisharp360")
        result = conv.convert(
            input_path=str(img), output_path=str(out),
            generator="unisharp360",
            unisharp_repo="/repo", unisharp_checkpoint="/ckpt.pt",
        )
        assert captured["backend"] == "unisharp"
        assert captured["unisharp_repo"] == "/repo"
        assert result.splat_count == 5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_dispatch.py::TestCoreRouting -v`
Expected: FAIL — `TypeError: convert() got an unexpected keyword argument 'unisharp_repo'`

- [ ] **Step 3: Extend the `convert` signature**

In `spag4d/core.py`, add these parameters to `SPAG4D.convert` after `pager_use_normals: bool = False,` (currently line 107):

```python
        pager_use_normals: bool = False,
        # --- new: sharp360 backend selection ---
        sharp_backend: str = "sharp",
        unisharp_repo: Optional[str] = None,
        unisharp_python: Optional[str] = None,
        unisharp_checkpoint: Optional[str] = None,
        unisharp_scale_align: str = "global",
        unisharp_format_mode: str = "copy",
        unisharp_save_debug: bool = False,
        unisharp_raw_output_dir: Optional[str] = None,
    ) -> ConversionResult:
```

- [ ] **Step 4: Extend the dispatch block**

Replace the existing `sharp360` dispatch (currently `spag4d/core.py:157-181`) — change the condition and the `convert_sharp360` call. The new block:

```python
        # Dispatch to sharp360 / unisharp360 generators if requested
        active_generator = generator or self.generator
        if active_generator in ("sharp360", "unisharp360"):
            from .sharp360 import convert_sharp360
            from .seedvr2 import SeedVR2Config
            seedvr2_cfg = SeedVR2Config() if seedvr2_upscale else None

            effective_backend = sharp_backend
            if active_generator == "unisharp360":
                effective_backend = "unisharp"

            result_dict = convert_sharp360(
                input_path=str(input_path),
                output_path=str(output_path),
                device=self.device,
                side_count=side_count,
                seedvr2_upscale=seedvr2_upscale,
                seedvr2_config=seedvr2_cfg,
                include_caps=sharp_include_caps,
                cap_fov_degrees=sharp_cap_fov,
                seam_latitude_degrees=sharp_seam_latitude,
                backend=effective_backend,
                unisharp_repo=unisharp_repo,
                unisharp_python=unisharp_python,
                unisharp_checkpoint=unisharp_checkpoint,
                unisharp_scale_align=unisharp_scale_align,
                unisharp_format_mode=unisharp_format_mode,
                unisharp_save_debug=unisharp_save_debug,
                unisharp_raw_output_dir=unisharp_raw_output_dir,
            )
            file_size = Path(output_path).stat().st_size
            return ConversionResult(
                output_path=str(output_path),
                splat_count=result_dict["num_gaussians"],
                file_size=file_size,
                processing_time=result_dict.get("processing_time", 0.0),
                depth_range=result_dict.get("depth_range", (0.0, 0.0)),
                panorama_size=(W, H),
            )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_dispatch.py::TestCoreRouting -v`
Expected: PASS

- [ ] **Step 6: Verify nothing else broke**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_dispatch.py tests/test_sharp360.py -v`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add spag4d/core.py tests/test_unisharp_dispatch.py
git commit -m "feat(core): route unisharp360 generator + thread backend args"
```

---

## Task 9: `cli.py` — Click options & generator choice

**Files:**
- Modify: `spag4d/cli.py` (`--generator` choices, new options, `convert()` params, `run_single` call)
- Test: `tests/test_unisharp_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_unisharp_cli.py`:

```python
"""Tests for the unisharp CLI surface."""
from click.testing import CliRunner

from spag4d.cli import convert


def test_help_lists_unisharp_options():
    runner = CliRunner()
    result = runner.invoke(convert, ["--help"])
    assert result.exit_code == 0
    assert "--sharp-backend" in result.output
    assert "--unisharp-repo" in result.output
    assert "--unisharp-checkpoint" in result.output
    assert "unisharp360" in result.output  # appears in --generator choices help


def test_generator_choice_accepts_unisharp360(tmp_path, monkeypatch):
    """Invoking with --generator unisharp360 reaches converter.convert with
    the unisharp backend args (we stub SPAG4D to avoid real inference)."""
    import numpy as np
    from PIL import Image
    import spag4d.core as core_mod

    img = tmp_path / "pano.jpg"
    Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8)).save(img)
    out = tmp_path / "out.ply"

    captured = {}

    class _StubConverter:
        def __init__(self, *a, **k):
            pass

        def convert(self, **kwargs):
            captured.update(kwargs)

            class _R:
                splat_count = 3
                file_size = 100
                processing_time = 0.0
                depth_range = (0.0, 0.0)
            return _R()

    monkeypatch.setattr(core_mod, "SPAG4D", _StubConverter)

    runner = CliRunner()
    result = runner.invoke(convert, [
        str(img), str(out),
        "--generator", "unisharp360",
        "--unisharp-repo", "/repo",
        "--unisharp-checkpoint", "/ckpt.pt",
    ])
    assert result.exit_code == 0, result.output
    assert captured["unisharp_repo"] == "/repo"
    assert captured["unisharp_checkpoint"] == "/ckpt.pt"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_cli.py -v`
Expected: FAIL — `--help` output lacks `--sharp-backend`; choice rejects `unisharp360`.

- [ ] **Step 3: Add `unisharp360` to the `--generator` choice**

In `spag4d/cli.py`, change the `--generator` option (currently line 40-41):

```python
@click.option('--generator', type=click.Choice(['da360', 'dap', 'sharp360', 'unisharp360', 'pager']),
              default=None, help='Generator mode: da360, dap, sharp360, unisharp360, or pager (overrides --depth-model)')
```

- [ ] **Step 4: Add the new Click options**

In `spag4d/cli.py`, add these options immediately after the `--seedvr2-upscale` option (currently line 50-51), before `def convert(`:

```python
@click.option('--sharp-backend', type=click.Choice(['sharp', 'unisharp', 'hybrid']),
              default='sharp', help='sharp360 backend (default: sharp)')
@click.option('--unisharp-repo', type=click.Path(), default=None,
              help='Path to a local clone of Insta360-Research-Team/UniSHARP')
@click.option('--unisharp-python', type=click.Path(), default=None,
              help='python executable of the unisharp conda env')
@click.option('--unisharp-checkpoint', type=click.Path(), default=None,
              help='UniSHARP checkpoint (step_XXXXXXX.pt)')
@click.option('--unisharp-scale-align', type=click.Choice(['none', 'global', 'da360_grid']),
              default='global', help='UniSHARP scale alignment mode (default: global)')
@click.option('--unisharp-format-mode', type=click.Choice(['copy', 'convert']),
              default='copy', help='UniSHARP PLY format handling (default: copy)')
@click.option('--unisharp-save-debug', is_flag=True,
              help='Keep UniSHARP raw PLY, gifs, and metadata')
@click.option('--unisharp-raw-output-dir', type=click.Path(), default=None,
              help='Persist the UniSHARP working dir here (default: temp dir)')
```

- [ ] **Step 5: Add the params to `convert()` and thread to `run_single`**

In `spag4d/cli.py`, add to the `convert(...)` function signature after `seedvr2_upscale: bool,` (currently line 75):

```python
    seedvr2_upscale: bool,
    sharp_backend: str,
    unisharp_repo: str,
    unisharp_python: str,
    unisharp_checkpoint: str,
    unisharp_scale_align: str,
    unisharp_format_mode: str,
    unisharp_save_debug: bool,
    unisharp_raw_output_dir: str,
):
```

Then add to the `converter.convert(...)` call inside `run_single` (currently ends at line 129 with `pager_use_normals=pager_use_normals,`):

```python
            pager_use_normals=pager_use_normals,
            sharp_backend=sharp_backend,
            unisharp_repo=unisharp_repo,
            unisharp_python=unisharp_python,
            unisharp_checkpoint=unisharp_checkpoint,
            unisharp_scale_align=unisharp_scale_align,
            unisharp_format_mode=unisharp_format_mode,
            unisharp_save_debug=unisharp_save_debug,
            unisharp_raw_output_dir=unisharp_raw_output_dir,
        )
```

- [ ] **Step 6: Update the mode banner (optional polish)**

In `spag4d/cli.py`, extend the mode echo (currently line 92) so unisharp360 reads clearly:

```python
        if generator == 'sharp360':
            mode = f"SHARP 360 (backend={sharp_backend}, sides={side_count}{', SeedVR2 upscale' if seedvr2_upscale else ''})"
        elif generator == 'unisharp360':
            mode = "UniSHARP 360 (native ERP)"
        elif sharp_refine:
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_unisharp_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 8: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS (no regressions; new unisharp tests green)

- [ ] **Step 9: Commit**

```bash
git add spag4d/cli.py tests/test_unisharp_cli.py
git commit -m "feat(cli): unisharp360 generator + --sharp-backend/--unisharp-* options"
```

---

## Task 10: M1.5 — Smoke test & viewer reality check (MANUAL, requires real env)

This task is the only one that needs a real UniSHARP install + GPU. It is **gated and manual** — do not block the code merge on it, but run it before relying on the backend. It also confirms the unverified `color_space` PLY encoding (see "Corrections" #4).

**Files:**
- Create: `docs/unisharp_smoke_runbook.md` (the runbook below, saved for reuse)

- [ ] **Step 1: One-time environment setup** (per design doc §11.1)

```bash
git clone https://github.com/Insta360-Research-Team/UniSHARP.git
cd UniSHARP
conda create -n unisharp python=3.12 -y
conda activate unisharp
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
git clone https://github.com/lpiccinelli-eth/UniK3D.git UniK3D
# download a checkpoint from HuggingFace Insta360-Research/Unisharp -> step_XXXXXXX.pt
```

- [ ] **Step 2: Run the SPAG4d smoke command** (use a real 2:1 ERP pano)

```bash
.venv\Scripts\python.exe -m spag4d convert test_pano.jpg test_unisharp.ply ^
  --generator unisharp360 ^
  --unisharp-repo D:\repos\UniSHARP ^
  --unisharp-python D:\envs\unisharp\python.exe ^
  --unisharp-checkpoint D:\models\unisharp\step_0100000.pt ^
  --unisharp-save-debug
```

Expected: command completes; `test_unisharp.ply` exists with > 0 vertices; `test_unisharp_unisharp_debug/` contains `raw_gaussians.ply`, `metadata.json`, and the two gifs.

- [ ] **Step 3: Confirm the `color_space` encoding**

Inspect the raw PLY header to see how UniSHARP actually stores `color_space` (comment vs. element vs. absent):

```bash
.venv\Scripts\python.exe -c "from plyfile import PlyData; p=PlyData.read('test_unisharp_unisharp_debug/raw_gaussians.ply'); print('comments:', p.comments); print('elements:', [e.name for e in p.elements])"
```

If `color_space` is stored as a PLY *element* rather than a comment, update `_read_color_space` in `spag4d/unisharp_format.py` to read it from that element, add a regression test to `tests/test_unisharp_format.py`, and commit. If it is a comment (as the fixture assumes) or absent, no change needed.

- [ ] **Step 4: Viewer reality check** (kills-or-confirms whether `convert` mode is needed)

Open `test_unisharp.ply` (copy mode output) in **SuperSplat** and SPAG4d's own GaussianSplats3D viewer. Check orientation, scale, color, opacity. Record findings in `docs/unisharp_smoke_runbook.md`:
- If the supplement elements load fine everywhere and orientation/color are acceptable → `copy` mode is sufficient; `convert` mode is an optional fallback.
- If a strict loader chokes on supplement elements → `--unisharp-format-mode convert` is the recommended default for that loader.

- [ ] **Step 5: Comparison pass** (optional, informative)

Run the same pano through `--generator da360`, `--generator sharp360 --sharp-backend sharp`, and `--generator unisharp360`; compare gaussian count, seam behavior, and pole (nadir/zenith) quality. Note results in the runbook to inform the future hybrid plan (M6).

- [ ] **Step 6: Commit the runbook**

```bash
git add docs/unisharp_smoke_runbook.md
git commit -m "docs(unisharp): M1.5 smoke + viewer reality-check runbook"
```

---

## Task 11: Docs — README + install helper note

**Files:**
- Modify: `README.md` (backend table / usage section — match existing structure)
- Modify: `CLAUDE.md` (Architecture + Development Commands — add the unisharp backend row)

- [ ] **Step 1: Add the UniSHARP backend to README usage**

Add a usage block to `README.md` near the existing `sharp360` examples (match the surrounding format):

```markdown
### UniSHARP native-ERP backend (optional)

Runs Insta360 UniSHARP in its own conda env via subprocess (no in-process import).
Requires a local UniSHARP clone, its conda python, and a checkpoint.

    python -m spag4d convert pano.jpg out.ply \
      --generator unisharp360 \
      --unisharp-repo D:/repos/UniSHARP \
      --unisharp-python D:/envs/unisharp/python.exe \
      --unisharp-checkpoint D:/models/unisharp/step_0100000.pt

Env fallbacks: `SPAG4D_UNISHARP_REPO`, `SPAG4D_UNISHARP_PYTHON`,
`SPAG4D_UNISHARP_CHECKPOINT`. Add `--unisharp-format-mode convert` for strict
PLY loaders, `--unisharp-save-debug` to keep raw PLY + gifs + metadata.
```

- [ ] **Step 2: Add the backend to CLAUDE.md**

In `CLAUDE.md` under "Development Commands", add:

```bash
# CLI conversion (UniSHARP native-ERP backend — subprocess, own env)
python -m spag4d convert input.jpg output.ply --generator unisharp360 \
  --unisharp-repo <clone> --unisharp-python <env-python> --unisharp-checkpoint <step.pt>
```

And add a Core Modules row noting `unisharp360.py` / `unisharp_adapter.py` / `unisharp_format.py` as the optional UniSHARP backend (subprocess-isolated).

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs(unisharp): README usage + CLAUDE.md backend notes"
```

---

## Deferred Work (separate follow-up plans)

These were intentionally **not** included — each needs live GPU iteration and a validated M1 baseline first:

- **M4 — Global scale alignment.** Single scalar matching median UniSHARP radius to DA360 median depth; apply to positions and add `log(s)` to `scale_*`. Needs the `scale_align` hook in `unisharp360.py` (currently a documented no-op) wired to a `global_scale_factor` helper + a DA360 depth pass on the same ERP. Plan after M1.5 confirms baseline scale mismatch magnitude.
- **M5 — DA360 grid alignment.** Per-gaussian local scale field by projecting centers back to ERP and sampling DA360 disparity. Real research effort; defer until `global` is validated.
- **M6 — Hybrid backend.** Run SHARP face pass + UniSHARP native pass; use UniSHARP as nadir/zenith/seam filler. The `"hybrid"` branch currently raises `NotImplementedError`. Scope around UniSHARP's real edge (pole/seam quality), not camera-travel range (see design doc §16).

---

## Self-Review (completed during planning)

**Spec coverage (M1/M1.5/M3):**
- §2.3 `--save-ply` required → Task 4 (`if save_ply: cmd.append("--save-ply")`) + test asserts presence. ✓
- §2.4 `--camera panorama` → Task 4 + test. ✓
- §2.6 PLY located by glob (slug ≠ stem) → Task 4 (`out.glob("**/gaussians.ply")`). ✓
- §2.8 supplement elements / color_space → Tasks 1-3 (inspect detects supplements; convert drops them; color_space read tolerant). ✓
- §4.1 pruned CLI (no `--unisharp-camera-json`, no `--unisharp-max-long-edge`) → Task 9 omits both. ✓
- §5.1 three new files (no `unisharp_camera.py`) → Tasks 1-6. ✓
- §6 adapter (`cwd=repo`, `--save-ply`) → Task 4. ✓
- §7 wrapper (env fallbacks, validation, work dir, format, debug, stats dict) → Tasks 5-6. ✓
- §8 format (inspect/count/copy/convert) → Tasks 1-3. ✓
- §10.1/10.2 sharp360 + core changes → Tasks 7-8 (corrected to preserve existing call args). ✓
- §12 error handling messages → Tasks 4-5 + tests assert the message text. ✓
- §13 testing plan (unit + smoke) → Tasks 1-9 (unit) + Task 10 (smoke/visual). ✓
- §14 M1/M1.5/M3 milestones → this plan; M4/M5/M6 deferred. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every code step shows complete code. The one deliberate runtime no-op (`scale_align`) is documented as such and deferred to M4. ✓

**Type consistency:** `CORE_VERTEX_FIELDS` identical in `unisharp_format.py` and the test fixture; `convert_unisharp360` keyword names (`checkpoint_path`, `scale_align`, `format_mode`, `raw_output_dir`) match exactly between `sharp360.py` dispatch (Task 7), the wrapper definition (Task 5), and the dispatch test (Task 7). `convert_sharp360` new kwargs match between `core.py` call (Task 8) and the signature (Task 7). CLI option dest names (`unisharp_repo`, etc.) match `convert()` params and the `converter.convert()` call (Task 9). ✓

---

## Execution Handoff

Once this plan is approved, two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Tasks 1-9 are independent enough to verify incrementally; Task 10 is manual; Task 11 is docs.
2. **Inline Execution** — execute tasks in this session with checkpoints for review.
