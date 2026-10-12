@echo off
rem Double-click to open the pose comparison launcher.
rem The window that stays open behind it shows progress messages and any error.
cd /d "%~dp0"
".venv\Scripts\python.exe" demo.py %*
if errorlevel 1 pause
