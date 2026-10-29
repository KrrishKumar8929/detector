@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Recognition-Service.ps1"
echo.
echo If the install succeeded, press any key to close this window.
pause >nul
