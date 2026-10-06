@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. The explanation is above.
  pause
  exit /b 1
)
endlocal
