# spag4d/da360_arch/__init__.py
"""
DA360 (Depth Anything 360) model architecture.

Wraps the DA360 model from Insta360 Research.
https://github.com/Insta360-Research-Team/DA360

DA360 uses Depth Anything V2's ViT-Large backbone with all DPT decoder
Conv2d layers replaced by ERPCircularConv2d for seamless 360° processing.
A Shift MLP on the ViT class token converts affine-invariant disparity
to scale-invariant disparity.
"""

import sys
import os
from pathlib import Path

DA360_DIR = Path(__file__).parent / "DA360"


def _ensure_da360_on_path():
    """Ensure DA360 directory is on sys.path."""
    if not DA360_DIR.exists():
        return

    da360_str = str(DA360_DIR)
    if da360_str in sys.path:
        sys.path.remove(da360_str)
    sys.path.insert(0, da360_str)


def _flush_conflicting_modules():
    """
    Purge cached 'networks' and 'depth_anything_v2' modules from sys.modules.

    DAP and DA360 both have their own 'networks' package. If DAP was loaded
    first, Python caches DAP's 'networks' module, causing DA360's
    'from networks.da360 import DA360' to fail. Flushing these entries
    forces a fresh import from the now-correct sys.path.
    """
    stale_prefixes = ('networks', 'depth_anything_v2')
    for key in list(sys.modules.keys()):
        for prefix in stale_prefixes:
            if key == prefix or key.startswith(prefix + '.'):
                del sys.modules[key]
                break


_ensure_da360_on_path()


def build_da360_model(encoder: str = 'vitl'):
    """
    Build DA360 model architecture.

    Args:
        encoder: DINOv2 encoder variant ('vits', 'vitb', 'vitl')

    Returns:
        nn.Module: DA360 model ready for weight loading
    """
    _ensure_da360_on_path()
    _flush_conflicting_modules()

    if not DA360_DIR.exists():
        raise ImportError(
            f"DA360 directory not found at {DA360_DIR}\n\n"
            "To set up DA360:\n"
            "1. Clone https://github.com/Insta360-Research-Team/DA360 "
            "into spag4d/da360_arch/DA360/\n"
            "2. Download DA360_large.pth weights from Google Drive\n"
        )

    original_cwd = os.getcwd()
    try:
        os.chdir(str(DA360_DIR))

        from networks.da360 import DA360

        model = DA360(
            equi_h=518,
            equi_w=1036,
            dinov2_encoder=encoder,
        )
        return model

    except ImportError as e:
        raise ImportError(
            f"Failed to import DA360 model: {e}\n"
            "Check that the DA360 repo is properly cloned into "
            "spag4d/da360_arch/DA360/"
        )
    finally:
        os.chdir(original_cwd)


def is_da360_available() -> bool:
    """Check if DA360 architecture is available."""
    return DA360_DIR.exists() and any(DA360_DIR.iterdir())
