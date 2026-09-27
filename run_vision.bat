@echo off
setlocal EnableExtensions
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PYTHONUTF8=1"
set "VLLM_NO_USAGE_STATS=1"
set "HF_HUB_OFFLINE=1"
set "OMP_NUM_THREADS=4"
if not exist ".venv\Scripts\python.exe" (
  echo Run scripts\setup_kev_vision.ps1 first.
  popd
  pause
  exit /b 1
)
echo Loading Kev Vision. Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" scripts\vision_playground.py --open %*
set "KEV_EXIT=%ERRORLEVEL%"
popd
if not "%KEV_EXIT%"=="0" if not "%KEV_NO_PAUSE%"=="1" pause
exit /b %KEV_EXIT%
