@echo off
rem One step: create the Python environment, install the packages of requirements.txt,
rem then download the model weights. Needs Python 3.11 and about 6 GB of free disk space.
rem Safe to run again: steps that are already done are skipped.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 goto with_pip

echo [1/3] Creating the Python environment with uv ...
if not exist ".venv\Scripts\python.exe" uv venv .venv --python 3.11 || goto failed
echo [2/3] Installing the packages listed in requirements.txt ...
uv pip install --python ".venv\Scripts\python.exe" --no-deps --index-strategy unsafe-best-match -r requirements.txt || goto failed
goto models

:with_pip
echo uv was not found, so pip is used instead.
echo [1/3] Creating the Python environment ...
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv || goto failed
echo [2/3] Installing the packages listed in requirements.txt ...
".venv\Scripts\python.exe" -m pip install --no-deps -r requirements.txt || goto failed

:models
echo [3/3] Downloading the model weights (about 0.9 GB) ...
".venv\Scripts\python.exe" scripts\download_models.py || goto failed
echo.
echo Done. Double-click run_demo.bat to start.
pause
exit /b 0

:failed
echo.
echo Setup stopped with an error. The messages above show which step failed.
pause
exit /b 1
