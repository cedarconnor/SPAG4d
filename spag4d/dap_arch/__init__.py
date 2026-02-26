# spag4d/dap_arch/__init__.py
"""
DAP (Depth Any Panoramas) model architecture.

Wraps the DAP model from Insta360 Research.
https://github.com/Insta360-Research-Team/DAP
"""

import sys
import os
from pathlib import Path

# Get the DAP directory path (resolved at import time so it's always absolute)
DAP_DIR = Path(__file__).parent / "DAP"


def _ensure_dap_on_path():
    """Ensure DAP directories are at the front of sys.path (idempotent)."""
    if not DAP_DIR.exists():
        return

    dap_str = str(DAP_DIR)
    if dap_str in sys.path:
        sys.path.remove(dap_str)
    sys.path.insert(0, dap_str)

    depth_metric_dir = DAP_DIR / "depth_anything_v2_metric"
    if depth_metric_dir.exists():
        dm_str = str(depth_metric_dir)
        if dm_str in sys.path:
            sys.path.remove(dm_str)
        sys.path.insert(0, dm_str)


def _flush_dap_module_cache():
    """
    Remove any stale/failed DAP module imports from sys.modules.

    Python caches failed module lookups. If another model (e.g. PanDA) was
    loaded first and the DAP path wasn't yet on sys.path when Python tried to
    resolve 'networks.*', the failed result gets cached. This purges those
    entries so a fresh import attempt uses the now-correct sys.path.
    """
    stale_prefixes = ('networks', 'depth_anything_v2_metric', 'depth_anything_v2', 'dap')
    for key in list(sys.modules.keys()):
        for prefix in stale_prefixes:
            if key == prefix or key.startswith(prefix + '.'):
                del sys.modules[key]
                break


# Eagerly add to path at import time (handles the common case)
_ensure_dap_on_path()


def build_dap_model(max_depth: float = 100.0):
    """
    Build DAP model architecture.

    Args:
        max_depth: Maximum depth in meters for metric output

    Returns:
        nn.Module: DAP model ready for weight loading
    """
    # Re-ensure path (survives uvicorn --reload re-imports)
    _ensure_dap_on_path()
    # Purge any stale cached imports that may have been poisoned before the
    # path was correctly set (e.g. PanDA was loaded as the default model first)
    _flush_dap_module_cache()

    original_cwd = os.getcwd()

    try:
        # Change to DAP directory so any internal relative-path config works
        os.chdir(str(DAP_DIR))

        from argparse import Namespace
        from networks.dap import DAP

        args = Namespace()
        args.midas_model_type = 'vitl'
        args.fine_tune_type = 'none'
        args.min_depth = 0.001
        args.max_depth = max_depth
        args.train_decoder = False

        model = DAP(args)
        return model

    except ImportError as e:
        raise ImportError(
            f"Failed to import DAP model: {e}\n\n"
            "Make sure the DAP repository is properly set up:\n"
            "1. Initialize the DAP submodule: git submodule update --init --recursive\n"
            "   (or clone https://github.com/Insta360-Research-Team/DAP into spag4d/dap_arch/DAP)\n"
            "2. Install DAP dependencies: pip install einops opencv-python\n"
            "3. Ensure depth_anything_v2_metric is available in DAP/\n"
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def is_dap_available() -> bool:
    """Check if DAP architecture is available."""
    try:
        build_dap_model()
        return True
    except (ImportError, Exception):
        return False
