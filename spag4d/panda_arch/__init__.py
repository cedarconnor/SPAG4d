# spag4d/panda_arch/__init__.py
"""
PanDA (Panoramic Depth Anything) model architecture.

Wraps the PanDA model from caozidong's CVPR 2025 paper.
https://github.com/caozidong/PanDA
"""

import sys
import os
from pathlib import Path

# Get the PanDA directory path
PANDA_DIR = Path(__file__).parent / "PanDA"

def _ensure_panda_on_path():
    """Ensure PanDA directories are at the front of sys.path (idempotent)."""
    if not PANDA_DIR.exists():
        return

    panda_str = str(PANDA_DIR)
    if panda_str in sys.path:
        sys.path.remove(panda_str)
    sys.path.insert(0, panda_str)

    depth_metric_dir = PANDA_DIR / "depth_anything_v2_metric"
    dap_metric_dir = Path(__file__).parent.parent / "dap_arch" / "DAP" / "depth_anything_v2_metric"
    
    target_metric_dir = depth_metric_dir if depth_metric_dir.exists() else dap_metric_dir
    if target_metric_dir.exists():
        dm_str = str(target_metric_dir)
        if dm_str in sys.path:
            sys.path.remove(dm_str)
        sys.path.insert(0, dm_str)

# Eagerly add to path at import time
_ensure_panda_on_path()

def build_panda_model(lora_rank: int = 4):
    """
    Build PanDA model architecture.

    Args:
        lora_rank: LoRA rank for adaptation layers (default: 4)

    Returns:
        nn.Module: PanDA model ready for weight loading
    """
    import os

    # Re-ensure path
    _ensure_panda_on_path()

    # Evict any 'networks.*' entries that DAP may have cached in sys.modules.
    # DAP's networks/dap.py is incompatible with PanDA's networks/panda.py —
    # Python won't re-search sys.path for a module already in sys.modules,
    # so we must clear the stale cache before re-importing.
    stale_prefixes = ('networks', 'depth_anything_v2_metric', 'depth_anything_v2', 'panda', 'dap')
    for key in list(sys.modules.keys()):
        for prefix in stale_prefixes:
            if key == prefix or key.startswith(prefix + '.'):
                del sys.modules[key]
                break

    # Save original working directory
    original_cwd = os.getcwd()

    try:
        # Change to PanDA directory so relative paths work
        os.chdir(str(PANDA_DIR))

        from argparse import Namespace

        # Import PanDA components
        from networks.panda import PanDA

        args = Namespace()
        args.midas_model_type = 'vitl'
        args.fine_tune_type = 'inference'  # Skip loading base weights (we load full checkpoint)
        args.min_depth = 0.01
        args.max_depth = 1.0  # Relative depth output
        args.lora = True
        args.train_decoder = True
        args.lora_rank = lora_rank

        model = PanDA(args)
        return model

    except ImportError as e:
        raise ImportError(
            f"Failed to import PanDA model: {e}\n\n"
            "Make sure the PanDA repository is properly set up:\n"
            "1. Clone PanDA: git clone https://github.com/caozidong/PanDA into spag4d/panda_arch/PanDA\n"
            "2. Install PanDA dependencies: pip install einops opencv-python\n"
            "3. Ensure depth_anything_v2_metric is available\n"
        )
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def is_panda_available() -> bool:
    """Check if PanDA architecture is available."""
    return PANDA_DIR.exists() and (PANDA_DIR / "networks" / "panda.py").exists()
