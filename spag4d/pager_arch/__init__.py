"""Vendored prs-eth/PaGeR (DA3 cubemap multi-view geometry model).

Code license: Apache-2.0 (see PaGeR/LICENSE).
Model weights (prs-eth/PaGeR, downloaded separately): CC BY-NC 4.0 — NON-COMMERCIAL
only (see PaGeR/LICENSE-MODEL). DA360/DAP remain the commercial-safe defaults.

Pinned upstream commit: 99188f256a5220c37ed6aacb25804ac1dca1589a

The upstream package uses absolute ``src.*`` imports, so its repo root must be on
sys.path for ``import src.pager`` to resolve. ``_ensure_on_path()`` does that lazily.
"""
from pathlib import Path
import sys

_ARCH_ROOT = Path(__file__).parent / "PaGeR"


def _ensure_on_path() -> None:
    """Expose the vendored PaGeR roots on sys.path.

    The upstream code mixes two import styles — ``from src.depth_anything_3...``
    (needs the repo root) and ``from depth_anything_3...`` (needs ``src/``) —
    because it expects an editable install. Add both so either resolves.
    """
    for p in (str(_ARCH_ROOT), str(_ARCH_ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_pager(checkpoint_path, cfg, device, weight_dtype=None):
    """Construct the upstream Pager inference wrapper.

    Mirrors ``da360_arch.build_da360_model`` in role.

    Args:
        checkpoint_path: HF repo id (e.g. ``"prs-eth/PaGeR"``) or local dir with
            ``model.safetensors`` + ``config.yaml``.
        cfg: model config (OmegaConf), typically loaded from the checkpoint's
            ``config.yaml`` — carries ``modalities``, ``face_size``, ``log_depth``.
        device: torch device.
        weight_dtype: optional torch dtype for weights (default float32).
    """
    import torch
    _ensure_on_path()
    from src.pager import Pager  # noqa: E402  (vendored module, needs sys.path)

    kwargs = {}
    if weight_dtype is not None:
        kwargs["weight_dtype"] = weight_dtype
    return Pager(checkpoint_path, cfg=cfg, device=device, **kwargs)
