@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM MANGAL DRISHTI - Windows One-Click Mission Control Launcher
REM Creator: Dharm Patel (https://github.com/DDharm007/Mangal-Drishti)
REM Usage: Double-click START_MANGAL_DRISHTI.bat  (or pass --check for preflight)
REM ============================================================================

set "PREFLIGHT_ONLY=0"
if /i "%~1"=="--check" set "PREFLIGHT_ONLY=1"

REM 1. Verify Windows OS and move cleanly to Repository Root
if not "%OS%"=="Windows_NT" (
    echo [ERROR] This launcher is designed strictly for Microsoft Windows.
    exit /b 1
)
cd /d "%~dp0"
set "REPO_ROOT=%CD%"
set "LOG_DIR=%REPO_ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\launcher.log"
set "BACKEND_LOG=%LOG_DIR%\backend.log"
set "FRONTEND_LOG=%LOG_DIR%\frontend.log"

echo ============================================================================ > "%LOG_FILE%"
echo MANGAL DRISHTI Launcher Run: %DATE% %TIME% >> "%LOG_FILE%"
echo Repository Root: %REPO_ROOT% >> "%LOG_FILE%"

echo.
echo ============================================================================
echo   MANGAL DRISHTI - Mars Terrain Intelligence Mission Control
echo   Decoding the Red Planet Through Artificial Intelligence
echo ============================================================================
echo [INFO] Repository root: "%REPO_ROOT%"

REM 2. Verify required repository folders and entry files
set "MISSING_REPO_FILES=0"
for %%F in ("src\api\main.py" "src\intelligence\engine.py" "configs\data_config.yaml" "requirements.txt" "frontend\package.json" "frontend\index.html") do (
    if not exist "%%~F" (
        echo [ERROR] Required repository file missing: %%~F
        echo [ERROR] Required repository file missing: %%~F >> "%LOG_FILE%"
        set "MISSING_REPO_FILES=1"
    )
)
if "%MISSING_REPO_FILES%"=="1" (
    echo [ERROR] Incomplete repository structure. Re-clone https://github.com/DDharm007/Mangal-Drishti
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 1
)

REM 3. Check Python (3.10+ required; prefer existing .venv)
set "SYS_PYTHON="
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set SYS_PYTHON="%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set SYS_PYTHON=python
    ) else (
        where py >nul 2>&1
        if not errorlevel 1 set SYS_PYTHON=py -3
    )
)

if not defined SYS_PYTHON (
    echo [PREREQUISITE MISSING] Python 3.10+ was not found on PATH and .venv does not exist.
    echo [ACTION] Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
    echo          Ensure "Add Python to PATH" is checked during installation.
    echo [ERROR] Python prerequisite missing >> "%LOG_FILE%"
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 2
)

%SYS_PYTHON% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" > "%LOG_DIR%\pyver.tmp" 2>nul
set "PY_VER="
if exist "%LOG_DIR%\pyver.tmp" set /p PY_VER=<"%LOG_DIR%\pyver.tmp"
del /q "%LOG_DIR%\pyver.tmp" >nul 2>&1

%SYS_PYTHON% -c "import sys; print(int(sys.version_info >= (3, 10)))" > "%LOG_DIR%\pyok.tmp" 2>nul
set "PY_OK=0"
if exist "%LOG_DIR%\pyok.tmp" set /p PY_OK=<"%LOG_DIR%\pyok.tmp"
del /q "%LOG_DIR%\pyok.tmp" >nul 2>&1

if not "%PY_OK%"=="1" (
    echo [PREREQUISITE ERROR] Unsupported Python version: %PY_VER%. Python 3.10+ is required.
    echo [ACTION] Download Python 3.12 from https://www.python.org/downloads/windows/
    echo [ERROR] Unsupported Python version: %PY_VER% >> "%LOG_FILE%"
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 2
)
echo [OK] Python runtime: %PY_VER%

REM 4. Check Node.js (v18+) and npm
where node >nul 2>&1
if errorlevel 1 (
    echo [PREREQUISITE MISSING] Node.js was not found on PATH.
    echo [ACTION] Install Node.js LTS v20+ from https://nodejs.org/en/download/
    echo [ERROR] Node.js prerequisite missing >> "%LOG_FILE%"
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 3
)
where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [PREREQUISITE MISSING] npm was not found on PATH. Reinstall Node.js LTS from https://nodejs.org/
    echo [ERROR] npm prerequisite missing >> "%LOG_FILE%"
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 3
)
for /f "delims=" %%N in ('node --version 2^>nul') do set "NODE_VER=%%N"
for /f "delims=" %%M in ('npm.cmd --version 2^>nul') do set "NPM_VER=%%M"
echo [OK] Node.js %NODE_VER% / npm %NPM_VER%

REM 5. Create .venv only if missing (preserve existing venv safely)
if not exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    echo [SETUP] Creating Python virtual environment at .venv ...
    %SYS_PYTHON% -m venv "%REPO_ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create Python .venv.
        echo [ERROR] venv creation failed >> "%LOG_FILE%"
        if "%PREFLIGHT_ONLY%"=="0" pause
        exit /b 4
    )
)
set "VENV_PY_RAW=%REPO_ROOT%\.venv\Scripts\python.exe"
set VENV_PY="%REPO_ROOT%\.venv\Scripts\python.exe"

REM 6. Check core Python dependencies and install only missing ones
%VENV_PY% -c "import torch, torchvision, cv2, PIL, yaml, numpy, fastapi, uvicorn, multipart" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Missing Python dependencies detected in .venv. Installing from requirements.txt ...
    %VENV_PY% -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
    %VENV_PY% -m pip install -r "%REPO_ROOT%\requirements.txt" fastapi uvicorn python-multipart >> "%LOG_FILE%" 2>&1
    %VENV_PY% -c "import torch, torchvision, cv2, PIL, yaml, numpy, fastapi, uvicorn, multipart" >nul 2>&1
    if errorlevel 1 (
        echo [DEPENDENCY ERROR] Failed to verify Python packages inside .venv. Inspect logs\launcher.log.
        if "%PREFLIGHT_ONLY%"=="0" pause
        exit /b 5
    )
)
echo [OK] Python dependencies verified inside .venv

REM 7. Check CUDA and GPU availability (CPU fallback supported cleanly)
%VENV_PY% -c "import torch; print(('CUDA ready: ' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU fallback active (CUDA unavailable)')" > "%LOG_DIR%\compute.tmp" 2>nul
set "COMPUTE_MSG=CPU fallback active"
if exist "%LOG_DIR%\compute.tmp" set /p COMPUTE_MSG=<"%LOG_DIR%\compute.tmp"
del /q "%LOG_DIR%\compute.tmp" >nul 2>&1
echo [COMPUTE] %COMPUTE_MSG%
echo [COMPUTE] %COMPUTE_MSG% >> "%LOG_FILE%"

REM 8. Check trained checkpoints without auto-downloading weights
set "HAS_DEFAULT_CKPT=0"
if exist "%REPO_ROOT%\experiments\deeplabv3plus\best_model.pth" set "HAS_DEFAULT_CKPT=1"
if exist "%REPO_ROOT%\experiments\unet\best_model.pth" set "HAS_DEFAULT_CKPT=1"
if "%HAS_DEFAULT_CKPT%"=="0" (
    echo [WARNING] No trained model checkpoints found under experiments\deeplabv3plus\best_model.pth or experiments\unet\best_model.pth.
    echo [ACTION]  Live inference requires placing best_model.pth in experiments\deeplabv3plus\ and/or experiments\unet\.
    echo           Historical reports and metrics remain viewable even without checkpoints.
    echo [WARNING] Missing model checkpoints >> "%LOG_FILE%"
) else (
    echo [OK] Trained segmentation checkpoints present locally
)

REM 9. Check frontend dependencies (install only when node_modules is absent or incomplete)
if not exist "%REPO_ROOT%\frontend\node_modules\vite" (
    echo [SETUP] Installing frontend Node dependencies in frontend\ ...
    pushd "%REPO_ROOT%\frontend"
    call npm.cmd install >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        popd
        echo [DEPENDENCY ERROR] npm install failed inside frontend\. Inspect logs\launcher.log.
        if "%PREFLIGHT_ONLY%"=="0" pause
        exit /b 6
    )
    popd
)
echo [OK] Frontend node_modules verified

REM 10. Check port availability (8000 for FastAPI, 5173 for Vite) to avoid duplicate instances
set "PORT8000_BUSY=0"
set "PORT5173_BUSY=0"
powershell -NoProfile -Command "$c1 = (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue); if ($c1) { exit 1 }"
if errorlevel 1 set "PORT8000_BUSY=1"
powershell -NoProfile -Command "$c2 = (Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue); if ($c2) { exit 1 }"
if errorlevel 1 set "PORT5173_BUSY=1"

if "%PORT8000_BUSY%"=="1" if "%PORT5173_BUSY%"=="1" (
    echo [INFO] Ports 8000 and 5173 are already active. Checking if MANGAL DRISHTI is already running...
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/health' -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
    if not errorlevel 1 (
        echo [OK] Existing MANGAL DRISHTI instance detected cleanly at http://127.0.0.1:5173
        if "%PREFLIGHT_ONLY%"=="0" start "" "http://127.0.0.1:5173"
        exit /b 0
    )
)

if "%PORT8000_BUSY%"=="1" (
    echo [PORT CONFLICT] Port 8000 is already occupied by another process. Stop that process before starting FastAPI.
    echo [ERROR] Port 8000 busy >> "%LOG_FILE%"
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 7
)
if "%PORT5173_BUSY%"=="1" (
    echo [PORT CONFLICT] Port 5173 is already occupied by another process. Stop that process before starting Vite.
    echo [ERROR] Port 5173 busy >> "%LOG_FILE%"
    if "%PREFLIGHT_ONLY%"=="0" pause
    exit /b 7
)

if "%PREFLIGHT_ONLY%"=="1" (
    echo [PREFLIGHT PASS] All prerequisites, dependencies, and ports 8000/5173 are ready.
    exit /b 0
)

REM 11. Start FastAPI backend and capture tracked PID
echo [START] Launching FastAPI Backend on http://127.0.0.1:8000 ...
for /f "delims=" %%P in ('powershell -NoProfile -Command "$p = Start-Process -FilePath '%VENV_PY_RAW%' -ArgumentList '-m','uvicorn','src.api.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory '%REPO_ROOT%' -RedirectStandardOutput '%BACKEND_LOG%' -RedirectStandardError '%LOG_DIR%\backend.err.log' -WindowStyle Hidden -PassThru; Write-Output $p.Id"') do set "BACKEND_PID=%%P"

echo [WAIT] Waiting for FastAPI health endpoint (/api/health)...
powershell -NoProfile -Command "$ok=$false; for($i=0; $i -lt 30; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if($r.StatusCode -eq 200){ $ok=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
    echo [ERROR] Backend failed to pass /api/health within 30 seconds. Check logs\backend.err.log.
    if defined BACKEND_PID taskkill /PID %BACKEND_PID% /T /F >nul 2>&1
    pause
    exit /b 8
)
echo [OK] FastAPI Backend Online (PID %BACKEND_PID%)

REM 12. Start Vite Frontend and wait until port 5173 responds
echo [START] Launching Vite Frontend on http://127.0.0.1:5173 ...
for /f "delims=" %%F in ('powershell -NoProfile -Command "$p = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev','--','--host','127.0.0.1','--port','5173' -WorkingDirectory '%REPO_ROOT%\frontend' -RedirectStandardOutput '%FRONTEND_LOG%' -RedirectStandardError '%LOG_DIR%\frontend.err.log' -WindowStyle Hidden -PassThru; Write-Output $p.Id"') do set "FRONTEND_PID=%%F"

echo [WAIT] Waiting for Frontend HTTP readiness...
powershell -NoProfile -Command "$ok=$false; for($i=0; $i -lt 25; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:5173/' -TimeoutSec 2; if($r.StatusCode -eq 200){ $ok=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
    echo [ERROR] Frontend failed to respond on http://127.0.0.1:5173 within 25 seconds. Check logs\frontend.err.log.
    if defined FRONTEND_PID taskkill /PID %FRONTEND_PID% /T /F >nul 2>&1
    if defined BACKEND_PID taskkill /PID %BACKEND_PID% /T /F >nul 2>&1
    pause
    exit /b 9
)
echo [OK] Vite Frontend Ready (PID %FRONTEND_PID%)

echo.
echo ============================================================================
echo   MANGAL DRISHTI IS ONLINE
echo   UI URL  : http://127.0.0.1:5173
echo   API URL : http://127.0.0.1:8000/docs
echo   Logs    : %LOG_DIR%
echo ============================================================================
echo.
start "" "http://127.0.0.1:5173"

echo Keep this launcher window open while using MANGAL DRISHTI.
echo Press any key in this window when finished to cleanly shut down both servers...
pause >nul

echo [SHUTDOWN] Stopping Frontend (PID %FRONTEND_PID%) and Backend (PID %BACKEND_PID%)...
if defined FRONTEND_PID taskkill /PID %FRONTEND_PID% /T /F >nul 2>&1
if defined BACKEND_PID taskkill /PID %BACKEND_PID% /T /F >nul 2>&1
echo [OK] Clean shutdown complete.
exit /b 0
