@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==================================================
echo   SPAG-4D v2.0 Installer
echo ==================================================
echo.

:: ──────────────────────────────────────────────────
:: Check for Git
:: ──────────────────────────────────────────────────
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not on your PATH.
    echo         Download it from: https://git-scm.com/downloads
    echo         After installing, restart this script.
    pause
    exit /b 1
)

:: ──────────────────────────────────────────────────
:: Embedded Python Setup
:: ──────────────────────────────────────────────────
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_ZIP=python_embed.zip"
set "PYTHON_DIR=python_embed"

if exist "%PYTHON_DIR%\python.exe" (
    echo [OK] Embedded Python already installed.
    goto :InstallDeps
)

echo [1/4] Downloading Python 3.11 Embedded...
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"
if not exist "%PYTHON_ZIP%" (
    echo [ERROR] Failed to download Python. Check your internet connection.
    pause
    exit /b 1
)

echo [2/4] Extracting Python...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%PYTHON_ZIP%"

echo [3/4] Configuring Embedded Python...
set "PTH_FILE=%PYTHON_DIR%\python311._pth"
powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site', 'import site' | Set-Content '%PTH_FILE%'"
powershell -Command "Add-Content -Path '%PTH_FILE%' -Value '..'"

echo [4/4] Installing pip...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"
"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py"
del "%PYTHON_DIR%\get-pip.py"

echo.

:: ──────────────────────────────────────────────────
:: Install Dependencies
:: ──────────────────────────────────────────────────
:InstallDeps
echo ==================================================
echo   Installing Dependencies
echo ==================================================
echo.

set "PIP=%PYTHON_DIR%\Scripts\pip.exe"
if not exist "!PIP!" set "PIP=%PYTHON_DIR%\python.exe -m pip"

echo [1/4] Installing PyTorch (CUDA 12.1)...
!PIP! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARN] PyTorch install had errors. Retrying...
    !PIP! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo [2/4] Installing SPAG-4D...
!PIP! install -r requirements.txt
!PIP! install -e ".[server,download]"

echo.
echo [3/4] Installing ML-SHARP (Apple)...
if not exist "ml-sharp\pyproject.toml" (
    echo    Cloning ML-SHARP...
    git clone https://github.com/apple/ml-sharp ml-sharp
)
if exist "ml-sharp\pyproject.toml" (
    !PIP! install --no-deps -e ml-sharp
    echo    [OK] ML-SHARP installed.
) else (
    echo    [ERROR] ML-SHARP clone failed. SHARP is required for SPAG-4D.
    echo           Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [4/4] Installing DAP depth model...
if not exist "spag4d\dap_arch\DAP\networks" (
    echo    Initializing DAP submodule...
    git submodule update --init --recursive
    if not exist "spag4d\dap_arch\DAP\networks" (
        echo    Submodule failed, cloning DAP manually...
        git clone https://github.com/Insta360-Research-Team/DAP spag4d\dap_arch\DAP
    )
)
echo    [OK] DAP ready.

echo.
echo ==================================================
echo   Installation Complete!
echo.
echo   Run 'run.bat' to start SPAG-4D.
echo   Opens http://localhost:7860 in your browser.
echo ==================================================
echo.
pause
