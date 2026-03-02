"""
SPAG-4D v2 Portable Release Builder

Creates a minimal zip for portable distribution:
- Source code (spag4d/, static/, api.py)
- Install and launch scripts (install.bat, run.bat)
- Documentation (README.md, INSTALL.md)
- Configuration (requirements.txt, pyproject.toml, .gitmodules)
- Demo image (TestImage/monbachtal_riverbank_primary.jpg)

The user extracts the zip, runs install.bat (downloads Python + deps),
then runs run.bat to launch.
"""

import os
import zipfile

VERSION = "2.0.0"
ZIP_NAME = f"SPAG-4D-v{VERSION}-portable.zip"
ROOT_IN_ZIP = "SPAG-4D"

# Directories to include (recursively)
INCLUDE_DIRS = [
    "spag4d",
    "static",
]

# Individual files to include
INCLUDE_FILES = [
    "api.py",
    "install.bat",
    "run.bat",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "INSTALL.md",
    ".gitmodules",
]

# Demo image (single file, not the whole TestImage dir)
DEMO_IMAGE = os.path.join("TestImage", "monbachtal_riverbank_primary.jpg")

# Patterns to skip inside included directories
SKIP_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".egg-info",
    ".DS_Store",
    "Thumbs.db",
]

# Subdirectories to skip entirely
SKIP_DIRS = [
    os.path.join("spag4d", "dap_arch", "DAP"),     # Git submodule, cloned at install
    os.path.join("spag4d", "panda_arch", "PanDA"),  # Removed in v2, but dir may linger
    os.path.join("spag4d", "panda_arch"),            # Entire panda_arch is dead code
]

# Specific files to skip
SKIP_FILES = [
    # Test splat files left from debugging
    os.path.join("static", "test_single_face.splat"),
    os.path.join("static", "test_single_face_raw.splat"),
    os.path.join("static", "test_single_face_srgb.splat"),
]


def should_skip(filepath):
    """Check if a file should be excluded from the release."""
    for pattern in SKIP_PATTERNS:
        if pattern in filepath:
            return True
    for skip_dir in SKIP_DIRS:
        if filepath.startswith(skip_dir + os.sep) or filepath == skip_dir:
            return True
    if filepath in SKIP_FILES:
        return True
    return False


def build_zip():
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    print(f"Building {ZIP_NAME}...")
    print()

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add individual files
        for f in INCLUDE_FILES:
            if os.path.exists(f):
                arcname = os.path.join(ROOT_IN_ZIP, f)
                zf.write(f, arcname)
                file_count += 1
                total_size += os.path.getsize(f)
                print(f"  + {f}")
            else:
                print(f"  ! {f} (not found, skipping)")

        # Add demo image
        if os.path.exists(DEMO_IMAGE):
            arcname = os.path.join(ROOT_IN_ZIP, DEMO_IMAGE)
            zf.write(DEMO_IMAGE, arcname)
            file_count += 1
            total_size += os.path.getsize(DEMO_IMAGE)
            print(f"  + {DEMO_IMAGE}")
        else:
            print(f"  ! {DEMO_IMAGE} (not found)")

        # Add directories
        for d in INCLUDE_DIRS:
            if not os.path.exists(d):
                print(f"  ! {d}/ (not found, skipping)")
                continue

            dir_files = 0
            for root, dirs, files in os.walk(d):
                dirs[:] = [
                    dd for dd in dirs
                    if not should_skip(os.path.join(root, dd))
                ]

                for file in files:
                    filepath = os.path.join(root, file)
                    if should_skip(filepath):
                        continue
                    arcname = os.path.join(ROOT_IN_ZIP, filepath)
                    zf.write(filepath, arcname)
                    dir_files += 1
                    total_size += os.path.getsize(filepath)

            file_count += dir_files
            print(f"  + {d}/ ({dir_files} files)")

        # DAP submodule placeholder
        dap_placeholder = os.path.join(
            ROOT_IN_ZIP, "spag4d", "dap_arch", "DAP", ".gitkeep"
        )
        zf.writestr(dap_placeholder, "# DAP submodule - populated by install.bat\n")

    zip_size = os.path.getsize(ZIP_NAME) / (1024 * 1024)
    print()
    print(f"Done: {ZIP_NAME}")
    print(f"  {file_count} files, {total_size / (1024*1024):.1f} MB uncompressed")
    print(f"  {zip_size:.1f} MB compressed")


if __name__ == "__main__":
    build_zip()
