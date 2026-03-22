"""
Bootstrap MSVC environment for gsplat JIT compilation on Windows.

Import this module before any gsplat imports to ensure the CUDA
extension can compile. Required because:
  1. VS 2019 cl.exe is not on PATH by default
  2. setuptools distutils doesn't register _msvccompiler as attribute
  3. CUDA 12.1 rejects VS 18+ compilers (only 2017-2022 supported)

Once gsplat is compiled (cached), this is a no-op on subsequent runs.
"""

import os
import sys


def setup_msvc_env():
    """Configure MSVC environment for CUDA JIT compilation."""
    if os.name != "nt":
        return  # Linux/Mac don't need this

    # Check if cl.exe is already accessible
    import shutil
    if shutil.which("cl"):
        return  # Already on PATH

    # Search for a CUDA-12.1-compatible MSVC (VS 2019 preferred, then 2022 Build Tools)
    candidates = [
        # VS 2019 Community (MSVC 14.29 = _MSC_VER 1929)
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        # VS 2022 Build Tools
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        # VS 2019 Build Tools
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
        # VS 2022 Community
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
    ]

    msvc_root = None
    for base in candidates:
        if os.path.isdir(base):
            versions = sorted(os.listdir(base))
            for v in versions:
                cl_path = os.path.join(base, v, "bin", "Hostx64", "x64", "cl.exe")
                if os.path.isfile(cl_path):
                    msvc_root = os.path.join(base, v)
                    break
        if msvc_root:
            break

    if not msvc_root:
        return  # No MSVC found — gsplat will fail with a clear error later

    # Find Windows SDK
    sdk_inc = r"C:\Program Files (x86)\Windows Kits\10\Include"
    sdk_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib"
    sdk_version = None
    if os.path.isdir(sdk_inc):
        versions = sorted(os.listdir(sdk_inc))
        for v in reversed(versions):
            if os.path.isdir(os.path.join(sdk_inc, v, "ucrt")):
                sdk_version = v
                break

    if not sdk_version:
        return

    # Set environment
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["PATH"] = os.path.join(msvc_root, "bin", "Hostx64", "x64") + ";" + os.environ.get("PATH", "")
    os.environ["INCLUDE"] = ";".join([
        os.path.join(msvc_root, "include"),
        os.path.join(sdk_inc, sdk_version, "ucrt"),
        os.path.join(sdk_inc, sdk_version, "shared"),
        os.path.join(sdk_inc, sdk_version, "um"),
    ])
    os.environ["LIB"] = ";".join([
        os.path.join(msvc_root, "lib", "x64"),
        os.path.join(sdk_lib, sdk_version, "ucrt", "x64"),
        os.path.join(sdk_lib, sdk_version, "um", "x64"),
    ])

    # Set CUDA arch for the current GPU
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{props.major}.{props.minor}"
        except Exception:
            pass

    # Fix setuptools distutils: force _msvccompiler to be importable as attribute
    try:
        import distutils
        import distutils._msvccompiler
        sys.modules["distutils._msvccompiler"] = distutils._msvccompiler
    except ImportError:
        pass


# Auto-setup on import
setup_msvc_env()
