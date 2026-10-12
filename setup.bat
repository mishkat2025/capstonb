@echo off
rem One step: create the Python environment, install the packages of requirements.txt,
rem then download the model weights.
rem
rem Nothing has to be installed beforehand. The packages are installed with uv. When this
rem computer does not have uv, it is downloaded into the .tools folder of this project,
rem and uv downloads Python 3.11 when this computer does not have that either.
rem Needs an internet connection and about 7 GB of free disk space.
rem Safe to run again: steps that are already done are skipped.
setlocal
cd /d "%~dp0"

set "UV=uv"
where uv >nul 2>nul && goto have_uv
set "UV=%~dp0.tools\uv.exe"
if exist "%UV%" goto have_uv

echo [0/3] uv, the tool that installs Python and the packages, is not on this computer.
echo       Downloading it into the .tools folder of this project (23 MB, from github.com/astral-sh/uv) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = 'Tls12'; New-Item -ItemType Directory -Force '.tools' | Out-Null; Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.11.2/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\uv.zip'; if ((Get-FileHash '.tools\uv.zip' -Algorithm SHA256).Hash -ne '171b7ccda1bbd562da6babeffcf533a1c6cc7862cf998da826e1db534fc43e48') { throw 'The downloaded file is not the one uv published (checksum differs). Nothing was installed.' }; Expand-Archive '.tools\uv.zip' '.tools' -Force; $exe = Get-ChildItem '.tools' -Recurse -Filter 'uv.exe' | Select-Object -First 1; if ($exe.FullName -ne (Join-Path (Resolve-Path '.tools') 'uv.exe')) { Copy-Item $exe.FullName '.tools\uv.exe' }"
if errorlevel 1 goto failed
if not exist "%UV%" goto failed

:have_uv
echo [1/3] Creating the Python environment (Python 3.11 is downloaded if this computer does not have it) ...
if exist ".venv\Scripts\python.exe" goto packages
"%UV%" venv .venv --python 3.11
if errorlevel 1 goto failed

:packages
echo [2/3] Installing the packages listed in requirements.txt (about 4 GB, this can take a while) ...
"%UV%" pip install --python ".venv\Scripts\python.exe" --no-deps --index-strategy unsafe-best-match -r requirements.txt
if errorlevel 1 goto failed

echo [3/3] Downloading the model weights (about 0.9 GB) ...
".venv\Scripts\python.exe" scripts\download_models.py
if errorlevel 1 goto failed

echo.
echo Done. Double-click run_demo.bat to start.
pause
exit /b 0

:failed
echo.
echo Setup stopped with an error. The messages above show which step failed.
echo It is safe to run setup.bat again: finished steps are skipped.
pause
exit /b 1
