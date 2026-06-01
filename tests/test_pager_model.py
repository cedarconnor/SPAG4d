"""PaGeRModel matches the depth-backend contract. Structural tests use a
monkeypatched upstream so they run without the ~5.7GB checkpoint or a GPU.

A weights-guarded end-to-end test lives at the bottom (skipped unless the
prs-eth/PaGeR weights are cached and CUDA is present)."""
from pathlib import Path

import numpy as np
import pytest
import torch


class _FakeCfg:
    face_size = 16
    cube_fov = 90.0
    log_depth = True
    modalities = ["depth", "normals", "sky", "scale_indoor", "scale_outdoor"]


class _FakePager:
    """Minimal stand-in for the vendored Pager: returns ERP tensors at orig_size."""

    def __init__(self):
        self.weight_dtype = torch.float32

    def get_intrinsics_extrinsics(self, **k):
        pass

    def set_depth_range(self, *a):
        pass

    class _M:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

    model = _M()

    def __call__(self, cubemap, dtype=None, skip_heads=None):
        # (B,6,C,h,w) raw per-face outputs
        return {
            "depth": torch.zeros(1, 6, 1, 8, 8),
            "normals": torch.zeros(1, 6, 3, 8, 8),
            "sky": torch.full((1, 6, 1, 8, 8), -10.0),  # logits -> sigmoid ~ 0 (no sky)
            "scale": torch.zeros(1, 1),
        }

    def process_depth_output(self, pred, orig_size, sky_mask=None, log_scale=None, **k):
        H, W = orig_size
        return torch.full((1, H, W), 5.0), None  # (1,H,W) ERP radial depth

    def process_normals_output(self, pred, orig_size, sky_mask=None, **k):
        H, W = orig_size
        n = torch.zeros(3, H, W)
        n[1] = 1.0  # +y unit normals, channel-first (3,H,W)
        return n


def _install_fakes(monkeypatch):
    from spag4d import pager_model
    H_face = 16

    monkeypatch.setattr(pager_model, "_load_cfg", lambda: _FakeCfg())
    monkeypatch.setattr(pager_model, "build_pager",
                        lambda *a, **k: _FakePager())
    monkeypatch.setattr(pager_model, "_erp_to_cubemap",
                        lambda chw, face_w, fov: torch.zeros(6, 3, 8, 8))
    # sky cube (6,1,8,8) prob -> ERP (1,H,W)
    monkeypatch.setattr(
        pager_model, "_cubemap_to_erp",
        lambda cube, H, W, fov: torch.zeros(1, H, W))


def test_predict_returns_depth_and_sky_and_stashes_normals(monkeypatch):
    from spag4d import pager_model
    _install_fakes(monkeypatch)

    H, W = 64, 128
    m = pager_model.PaGeRModel(_FakePager(), _FakeCfg(),
                               torch.device("cpu"), metric=False)
    image = torch.zeros(H, W, 3, dtype=torch.uint8)
    depth, sky = m.predict(image)

    assert depth.shape == (H, W)
    assert sky.shape == (H, W) and sky.dtype == torch.bool
    assert m.last_normals.shape == (H, W, 3)
    assert m.depth_convention == "radial"
    assert m.metric is False
    # native_resolution recorded as the per-face effective ERP (2*face, 4*face)
    assert m.native_resolution == (2 * 16, 4 * 16)
    # normals re-normalized to unit length after bilinear stitch
    norms = m.last_normals.reshape(-1, 3).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_predict_float_image_path(monkeypatch):
    from spag4d import pager_model
    _install_fakes(monkeypatch)
    m = pager_model.PaGeRModel(_FakePager(), _FakeCfg(),
                               torch.device("cpu"), metric=False)
    depth, sky = m.predict(torch.zeros(32, 64, 3, dtype=torch.float32))
    assert depth.shape == (32, 64)


# ── Weights-guarded end-to-end smoke (Task 10) ──────────────────────────────

def _weights_present():
    from spag4d.pager_model import PAGER_CACHE_DIR
    return Path(PAGER_CACHE_DIR).exists() and any(
        Path(PAGER_CACHE_DIR).rglob("*.safetensors"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _weights_present(), reason="PaGeR weights not downloaded")
def test_pager_end_to_end_depth_shape():
    from PIL import Image
    from spag4d.pager_model import PaGeRModel

    data = sorted(Path("tests/data").glob("*.jpg")) + sorted(Path("tests/data").glob("*.png"))
    if not data:
        pytest.skip("no test panorama in tests/data")
    img = np.array(Image.open(data[0]).convert("RGB"))
    H, W = img.shape[:2]
    m = PaGeRModel.load(device=torch.device("cuda"))
    depth, sky = m.predict(torch.from_numpy(img).cuda())
    assert depth.shape == (H, W)
    assert sky.shape == (H, W) and sky.dtype == torch.bool
    assert m.last_normals.shape == (H, W, 3)
