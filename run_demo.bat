@echo off
rem Double-click to open the pose comparison launcher.
rem The window that stays open behind it shows progress messages and any error.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
echo The Python environment has not been set up yet. Double-click setup.bat first.
pause
exit /b 1

:run
".venv\Scripts\python.exe" demo.py %*
if errorlevel 1 pause
