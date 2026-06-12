@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==================================================
echo   SPAG-4D v3.0 Installer
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

echo [1/7] Installing PyTorch (CUDA 12.1)...
!PIP! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARN] PyTorch install had errors. Retrying...
    !PIP! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo [2/7] Installing SPAG-4D dependencies...
!PIP! install -r requirements.txt
!PIP! install -e ".[server,download]"

echo.
echo [4/7] Installing DAP depth model...
if not exist "spag4d\dap_arch\DAP\networks" (
    echo    Initializing DAP submodule...
    git submodule update --init --recursive
    if not exist "spag4d\dap_arch\DAP\networks" (
        echo    Submodule failed, cloning DAP manually...
        git clone https://github.com/Insta360-Research-Team/DAP spag4d\dap_arch\DAP
    )
)
echo    [OK] DAP architecture ready.

echo.
echo [5/7] Installing DA360 depth model...
if not exist "spag4d\da360_arch\DA360\networks" (
    echo    Cloning DA360...
    git clone https://github.com/Insta360-Research-Team/DA360 spag4d\da360_arch\DA360
)
if exist "spag4d\da360_arch\DA360\networks" (
    echo    [OK] DA360 architecture ready.
) else (
    echo    [WARN] DA360 clone failed. DA360 depth model will not be available.
    echo          DAP (the default depth model) still works.
)

echo.
echo [6/9] Installing gdown (for DA360 weights)...
!PIP! install gdown

echo.
echo [7/9] Installing refinement dependencies (diffusers, transformers)...
!PIP! install "diffusers>=0.37.0" transformers accelerate peft gsplat

echo.
echo [+] Installing optional PaGeR backend dependencies (non-commercial)...
!PIP! install -r requirements-pager.txt
if errorlevel 1 (
    echo    [WARN] PaGeR deps failed; the optional "--generator pager" backend won't work.
) else (
    echo    [OK] PaGeR deps installed. Weights are NOT downloaded by default
    echo         ^(prs-eth/PaGeR, ~5.7 GB, CC BY-NC 4.0 non-commercial^).
    echo         To enable it:  python -m spag4d download-models --model pager
)

echo.
echo [8/9] Installing Python dev headers for gsplat CUDA compilation...
"%PYTHON_DIR%\python.exe" -c "import urllib.request,zipfile,io,os; url='https://www.nuget.org/api/v2/package/python/3.11.9'; print('Downloading Python 3.11.9 headers...'); data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SPAG4D'})).read(); z=zipfile.ZipFile(io.BytesIO(data)); inc=r'%PYTHON_DIR%\Include'; lib=r'%PYTHON_DIR%\libs'; os.makedirs(inc,exist_ok=True); os.makedirs(lib,exist_ok=True); [open(os.path.join(inc,n[len('tools/include/'):]),'wb').write(z.read(n)) for n in z.namelist() if n.startswith('tools/include/') and not n.endswith('/')]; [open(os.path.join(lib,n[len('tools/libs/'):]),'wb').write(z.read(n)) for n in z.namelist() if n.startswith('tools/libs/') and not n.endswith('/')]; print('Done: headers + libs installed')"
if errorlevel 1 (
    echo    [WARN] Python dev headers download failed. Refinement may not work.
) else (
    echo    [OK] Python headers and libs installed for CUDA compilation.
)

:: Fix gsplat Windows MSVC compatibility
"%PYTHON_DIR%\python.exe" -c "import os; p=os.path.join(r'%PYTHON_DIR%','Lib','site-packages','gsplat','cuda','_backend.py'); f=open(p).read(); open(p,'w').write(f.replace(\"extra_cflags = [opt_level, '-Wno-attributes']\",\"extra_cflags = [opt_level] if os.name == 'nt' else [opt_level, '-Wno-attributes']\"))" 2>nul
echo    [OK] gsplat MSVC compatibility patched.

echo.
echo [9/9] Downloading model weights...
echo    This may take several minutes on first install.
echo.

:: Set environment variable so models cache in a known location
set "SPAG4D_CACHE=%USERPROFILE%\.cache\spag4d"
if not exist "%SPAG4D_CACHE%" mkdir "%SPAG4D_CACHE%"

:: Download DAP weights (~1.5 GB)
if exist "%SPAG4D_CACHE%\model.pth" (
    echo    [OK] DAP weights already cached.
) else (
    echo    Downloading DAP weights (~1.5 GB)...
    "%PYTHON_DIR%\python.exe" -c "from spag4d.dap_model import DAPModel; DAPModel._get_or_download_weights()"
    if exist "%SPAG4D_CACHE%\model.pth" (
        echo    [OK] DAP weights downloaded.
    ) else (
        echo    [WARN] DAP weight download failed. Will retry on first use.
    )
)

:: Download DA360 weights (~1.3 GB)
if exist "%SPAG4D_CACHE%\DA360_large.pth" (
    echo    [OK] DA360 weights already cached.
) else (
    echo    Downloading DA360 weights (~1.3 GB)...
    "%PYTHON_DIR%\python.exe" -c "import gdown, os; cache=os.path.expanduser('~/.cache/spag4d'); os.makedirs(cache, exist_ok=True); gdown.download_folder('https://drive.google.com/drive/folders/1FMLWZfJ_IPKOa_cEbVqrq8_BRkl3oB_2', output=cache, quiet=False)"
    if exist "%SPAG4D_CACHE%\DA360_large.pth" (
        echo    [OK] DA360 weights downloaded.
    ) else (
        echo    [WARN] DA360 weight download failed. Will retry on first use.
    )
)

:: ──────────────────────────────────────────────────
:: Optional: ArtiFixer3D refine backend (WSL2 + Docker)
:: ──────────────────────────────────────────────────
echo.
echo ==================================================
echo   Optional: ArtiFixer3D refine backend (advanced)
echo ==================================================
echo   Higher-quality disocclusion repair via a 14B model that runs in
echo   WSL2 + Docker. OPTIONAL -- the SPAG core and the default OmniRoam
echo   refine do not need any of this.
echo.
echo   Requirements: WSL2 + Docker + NVIDIA Container Toolkit, an A6000-class
echo   GPU (~48 GB VRAM), and ~140 GB disk (57 GB image + 67 GB checkpoint +
echo   23 GB Wan mirror). Full steps are in INSTALL.md.
echo.
set "WANT_AF3D=N"
set /p WANT_AF3D="Configure the ArtiFixer3D backend now? (y/N): "
if /i not "!WANT_AF3D!"=="y" (
    echo    [SKIP] ArtiFixer3D not configured. See INSTALL.md to enable it later.
    goto :Done
)

echo    Checking WSL2...
wsl -l -q >nul 2>&1
if errorlevel 1 (
    echo    [SKIP] WSL2 not detected. Run 'wsl --install' first, then see INSTALL.md.
    goto :Done
)
echo    Checking Docker inside WSL (distro: Ubuntu)...
wsl -d Ubuntu bash -c "docker info" >nul 2>&1
if errorlevel 1 (
    echo    [SKIP] Docker not reachable in WSL. Start Docker / see INSTALL.md.
    goto :Done
)
echo    [OK] WSL2 + Docker detected.

:: Translate this repo dir to its WSL mount path, then copy the hydra wrapper
:: configs into the ArtiFixer repo (the one step the installer can automate).
:: Compute /mnt/<drive>/<path> in PowerShell — passing a Windows path through
:: wsl.exe eats single backslashes, so do the conversion on the Windows side.
set "REPO_DIR=%~dp0"
if "!REPO_DIR:~-1!"=="\" set "REPO_DIR=!REPO_DIR:~0,-1!"
set "REPO_WSL="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$p=$env:REPO_DIR; '/mnt/'+$p.Substring(0,1).ToLower()+$p.Substring(2).Replace('\','/')"`) do set "REPO_WSL=%%i"
wsl -d Ubuntu bash -c "test -d $HOME/ArtiFixer/thirdparty/3DGRUT-ArtiFixer/configs" >nul 2>&1
if errorlevel 1 (
    echo    [INFO] ArtiFixer repo not found at ~/ArtiFixer.
    echo           Clone it with submodules and build artifixer:cuda12 first; see INSTALL.md.
) else (
    wsl -d Ubuntu bash -c "cp '!REPO_WSL!/spag4d/refine/artifixer3d_resources/_artifixer_run.yaml' '!REPO_WSL!/spag4d/refine/artifixer3d_resources/_artifixer_distill.yaml' $HOME/ArtiFixer/thirdparty/3DGRUT-ArtiFixer/configs/" >nul 2>&1
    if errorlevel 1 (
        echo    [WARN] Could not copy hydra wrapper configs. Copy them manually; see INSTALL.md.
    ) else (
        echo    [OK] Hydra wrapper configs copied into the ArtiFixer repo.
    )
)
echo.
echo    Remaining one-time steps (see INSTALL.md "ArtiFixer3D backend"):
echo      1. Clone ArtiFixer + submodules to ~/ArtiFixer, build the artifixer:cuda12 image
echo      2. Place artifixer-14b.pt in /data/artifixer-checkpoints/
echo      3. Run once in WSL:  bash spag4d/refine/artifixer3d_resources/setup_wan_mirror.sh
echo      Then refine with backend=artifixer3d via POST /api/refine_v2; see README.

:Done
echo.
echo ==================================================
echo   Installation Complete!
echo.
echo   Run 'run.bat' to start SPAG-4D.
echo   Opens http://localhost:7860 in your browser.
echo.
echo   Depth models: DAP + DA360 (+ optional PaGeR, non-commercial)
echo   Refine: OmniRoam (default, WSL2) + optional ArtiFixer3D (WSL2/Docker)
echo ==================================================
echo.
pause
