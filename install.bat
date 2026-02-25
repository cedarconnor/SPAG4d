@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==================================================
echo   SPAG-4D Automatic Installer (Embedded Python)
echo ==================================================
echo.

set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_ZIP=python_embed.zip"
set "PYTHON_DIR=python_embed"

:: Check if already installed
if exist "%PYTHON_DIR%\python.exe" (
    echo [INFO] Embedded Python already found in "%PYTHON_DIR%".
    goto :InstallDeps
)

echo [1/4] Downloading Python 3.11 Embedded...
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"
if not exist "%PYTHON_ZIP%" (
    echo [ERROR] Failed to download Python.
    exit /b 1
)

echo [2/4] Extracting Python...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%PYTHON_ZIP%"

echo [3/4] Configuring Embedded Python...
:: Uncomment the import site line in the ._pth file to allow pip to work
set "PTH_FILE=%PYTHON_DIR%\python311._pth"
powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site', 'import site' | Set-Content '%PTH_FILE%'"
:: Add the parent directory so app imports (like api.py) work
powershell -Command "Add-Content -Path '%PTH_FILE%' -Value '..'"

echo [4/4] Installing pip...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"
"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py"
del "%PYTHON_DIR%\get-pip.py"

:InstallDeps
echo.
echo ==================================================
echo   Installing Dependencies...
echo ==================================================

set "PIP=%PYTHON_DIR%\Scripts\pip.exe"
if not exist "!PIP!" (
    :: Fallback if pip goes to the root folder
    set "PIP=%PYTHON_DIR%\python.exe -m pip"
)

echo.
echo [1/4] Installing PyTorch (CUDA 12.1)...
!PIP! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/4] Installing SPAG-4D Core Requirements...
!PIP! install -r requirements.txt
!PIP! install -e ".[server,download]"

echo.
echo [3/4] Installing SHARP Model...
if not exist "ml-sharp\pyproject.toml" (
    echo Cloning SHARP manually...
    git clone https://github.com/apple/ml-sharp ml-sharp
)
!PIP! install --no-deps -e ml-sharp

echo.
echo [4/5] Installing Depth Anything V3 Model...
!PIP! install hatchling moviepy^<2.0.0 pycolmap trimesh evo
if not exist "src\depth-anything-3\pyproject.toml" (
    echo Cloning Depth Anything V3 manually...
    git clone https://github.com/ByteDance-Seed/depth-anything-3 "src\depth-anything-3"
)
!PIP! install --no-deps --no-build-isolation -e "src\depth-anything-3"

echo.
echo [5/6] Installing Apple Depth Pro (optional - Phase 1 high-quality depth)...
if not exist "src\ml-depth-pro\pyproject.toml" (
    echo Cloning Apple Depth Pro...
    git clone https://github.com/apple/ml-depth-pro "src\ml-depth-pro"
)
if exist "src\ml-depth-pro\pyproject.toml" (
    !PIP! install --no-deps -e "src\ml-depth-pro"
    echo [INFO] Apple Depth Pro installed successfully.
    
    echo Downloading Depth Pro Weights...
    if not exist "checkpoints\depth_pro.pt" (
        mkdir checkpoints 2>nul
        powershell -Command "Invoke-WebRequest -Uri 'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt' -OutFile 'checkpoints\depth_pro.pt'"
    )
) else (
    echo [WARN] Apple Depth Pro clone failed. Skipping - Phase 1 depth fusion will be unavailable.
)

echo.
echo [6/6] Installing DAP Model...
!PIP! install einops opencv-python
if not exist "spag4d\dap_arch\DAP\networks" (
    echo Initializing DAP submodule...
    git submodule update --init --recursive
    if not exist "spag4d\dap_arch\DAP\networks" (
        echo Git submodule failed. Cloning DAP manually...
        git clone https://github.com/Insta360-Research-Team/DAP spag4d\dap_arch\DAP
    )
)
echo ==================================================
echo   Installation Complete!
echo   You can now double-click 'run.bat' to start!
echo ==================================================
echo Done
 