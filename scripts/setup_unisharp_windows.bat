@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo ==================================================
echo   SPAG-4D : UniSHARP 360 backend setup (native Windows)
echo ==================================================
echo.
echo   Sets up Insta360 UniSHARP as an isolated, out-of-process backend.
echo   UniK3D's KNN CUDA op is eval-only (inference degrades gracefully) and
echo   gsplat is never compiled because inference runs with --no-render, so
echo   this works on native Windows without a Linux/WSL toolchain.
echo.
echo   Requirements: git, an NVIDIA GPU + driver, ~10 GB disk
echo   (torch ~3 GB + checkpoint 4.7 GB). Python 3.11 is used if found.
echo.

:: ---- configurable locations (override via env before running) ----
if not defined UNISHARP_REPO   set "UNISHARP_REPO=D:\repos\UniSHARP"
if not defined UNISHARP_VENV   set "UNISHARP_VENV=D:\envs\unisharp"
if not defined UNISHARP_MODELS set "UNISHARP_MODELS=D:\models\unisharp"
set "CKPT=%UNISHARP_MODELS%\pretained_model.pt"

echo   Repo:       %UNISHARP_REPO%
echo   Venv:       %UNISHARP_VENV%
echo   Checkpoint: %CKPT%
echo.

where git >nul 2>&1 || (echo [ERROR] git not on PATH. & pause & exit /b 1)

:: ---- locate a Python 3.11 interpreter ----
set "BASEPY="
for /f "delims=" %%p in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "BASEPY=%%p"
if not defined BASEPY if exist "C:\Program Files\Python311\python.exe" set "BASEPY=C:\Program Files\Python311\python.exe"
if not defined BASEPY (
    echo [ERROR] Python 3.11 not found. Install it from python.org, then re-run.
    pause & exit /b 1
)
echo [OK] Base Python: %BASEPY%

:: ---- create the isolated venv ----
if exist "%UNISHARP_VENV%\Scripts\python.exe" (
    echo [OK] Venv already exists.
) else (
    echo [1/6] Creating venv...
    "%BASEPY%" -m venv "%UNISHARP_VENV%" || (echo [ERROR] venv creation failed. & pause & exit /b 1)
)
set "VPY=%UNISHARP_VENV%\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip >nul

:: ---- torch 2.8 (CUDA 12.8 wheels) ----
echo [2/6] Installing torch 2.8.0 + torchvision (cu128)...
"%VPY%" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 || (echo [ERROR] torch install failed. & pause & exit /b 1)

:: ---- clone UniSHARP + UniK3D (UniK3D must live INSIDE UniSHARP) ----
echo [3/6] Cloning UniSHARP + UniK3D...
if not exist "%UNISHARP_REPO%\scripts\infer_unisharp.py" (
    git clone https://github.com/Insta360-Research-Team/UniSHARP.git "%UNISHARP_REPO%" || (echo [ERROR] UniSHARP clone failed. & pause & exit /b 1)
)
if not exist "%UNISHARP_REPO%\UniK3D\unik3d" (
    git clone https://github.com/lpiccinelli-eth/UniK3D.git "%UNISHARP_REPO%\UniK3D" || (echo [ERROR] UniK3D clone failed. & pause & exit /b 1)
)

:: ---- dependencies (skip Linux-only triton; gsplat installs as a pure-python
::      wheel and is never compiled thanks to --no-render; add wandb for UniK3D) ----
echo [4/6] Installing UniSHARP dependencies (skipping triton; adding wandb)...
findstr /v /i /c:"triton" /c:"torch==" /c:"torchvision==" /c:"torchaudio==" "%UNISHARP_REPO%\requirements.txt" > "%UNISHARP_REPO%\requirements.windows.txt"
"%VPY%" -m pip install -r "%UNISHARP_REPO%\requirements.windows.txt" || (echo [ERROR] dependency install failed. & pause & exit /b 1)
"%VPY%" -m pip install wandb

:: ---- patch infer_unisharp.py for --no-render (PLY-only, no gsplat build) ----
echo [5/6] Patching infer_unisharp.py for --no-render...
"%BASEPY%" "%~dp0patch_unisharp_no_render.py" "%UNISHARP_REPO%"

:: ---- checkpoint (4.7 GB) ----
echo [6/6] Downloading UniSHARP checkpoint (4.7 GB, first run only)...
if exist "%CKPT%" (
    echo [OK] Checkpoint already present.
) else (
    if not exist "%UNISHARP_MODELS%" mkdir "%UNISHARP_MODELS%"
    "%VPY%" -c "from huggingface_hub import hf_hub_download; import shutil,os; p=hf_hub_download('Insta360-Research/Unisharp','pretained_model.pt'); shutil.copy(p, r'%CKPT%'); print('checkpoint ->', r'%CKPT%')" || echo [WARN] checkpoint download failed; download pretained_model.pt from huggingface.co/Insta360-Research/Unisharp manually.
)

echo.
echo ==================================================
echo   UniSHARP 360 backend ready.
echo.
echo   Start the server with these environment variables set so the web UI's
echo   "UniSHARP 360" generator and the CLI can find the backend:
echo.
echo     set SPAG4D_UNISHARP_REPO=%UNISHARP_REPO%
echo     set SPAG4D_UNISHARP_PYTHON=%VPY%
echo     set SPAG4D_UNISHARP_CHECKPOINT=%CKPT%
echo     set SPAG4D_UNISHARP_NO_RENDER=1
echo.
echo   Then: python -m spag4d serve --port 7860
echo ==================================================
echo.
pause
