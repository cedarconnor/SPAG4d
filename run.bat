@echo off
cd /d "%~dp0"

echo ========================================
echo   SPAG-4D Launcher
echo ========================================
echo.

REM Prefer .venv Python (has pre-compiled gsplat CUDA kernels for refinement)
REM Falls back to python_embed if .venv doesn't exist
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "python_embed\python.exe" (
    set "PYTHON_EXE=python_embed\python.exe"
) else (
    echo [ERROR] No Python found. Need .venv or python_embed.
    echo Please run 'install.bat' first!
    pause
    exit /b 1
)

echo Starting web server on http://localhost:7860
echo (any existing server on this port will be killed automatically)
echo Press Ctrl+C to stop
echo.

REM Open browser after short delay
start "" /min cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:7860"

"%PYTHON_EXE%" -m spag4d.cli serve --port 7860

pause
