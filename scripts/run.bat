@echo off
REM ------------------------------------------------------------------
REM run.bat - one-command launcher for Windows (cmd.exe).
REM
REM Creates a .venv on first run, installs requirements.txt, then
REM launches Streamlit. Honours %APP_PORT% (default 8501) and
REM %APP_ADDRESS% (default localhost).
REM
REM Usage:
REM   scripts\run.bat                 REM normal
REM   scripts\run.bat --server.port 9000
REM ------------------------------------------------------------------
setlocal ENABLEDELAYEDEXPANSION

REM ---- Repo root = parent of this script -----------------------------
pushd "%~dp0\.." >nul
set "REPO_ROOT=%CD%"
popd >nul
cd /d "%REPO_ROOT%"

REM ---- Resolve Python 3.10+ -----------------------------------------
set "PYTHON_BIN="
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_BIN=py -3.11"
    goto :have_python
)
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_BIN=python"
    goto :have_python
)
echo [run.bat] ERROR: no python or py on PATH. 1>&2
exit /b 1

:have_python

REM ---- Create venv on first run --------------------------------------
set "VENV_DIR=%REPO_ROOT%\.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [run.bat] Creating virtualenv at %VENV_DIR% ...
    %PYTHON_BIN% -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

REM ---- Install requirements (idempotent) -----------------------------
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

REM ---- Launch --------------------------------------------------------
if "%APP_PORT%"=="" set "APP_PORT=8501"
if "%APP_ADDRESS%"=="" set "APP_ADDRESS=localhost"
if "%APP_HEADLESS%"=="" set "APP_HEADLESS=false"
if "%APP_THEME%"=="" set "APP_THEME=light"
if "%LOG_LEVEL%"=="" set "LOG_LEVEL=INFO"

set "URL=http://%APP_ADDRESS%:%APP_PORT%"
echo [run.bat] Launching Streamlit on %URL%
echo [run.bat] (Stop the server with Ctrl-C.)
echo [run.bat] APP_THEME=%APP_THEME% LOG_LEVEL=%LOG_LEVEL%

streamlit run app.py ^
    --server.port %APP_PORT% ^
    --server.address %APP_ADDRESS% ^
    --server.headless %APP_HEADLESS% %*

endlocal
