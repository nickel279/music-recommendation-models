@echo off
cd /d "%~dp0"

if not exist .env.local (
  echo Missing .env.local.
  echo Copy .env.local.example to .env.local and fill in your keys.
  pause
  exit /b 1
)

set USE_NOTEBOOK_MODEL_BRIDGE=1
set PORT=5001
set MODEL_DATA_DIR=model-data-clean
set NOTEBOOK_MODEL_DATA_DIR=model-data-clean

echo Starting Music Eval with exported notebook models...
echo.
echo Site: http://localhost:5001
echo.
start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:5001'"

if exist "C:\Users\nadez\anaconda3\envs\surprise_env\python.exe" (
  "C:\Users\nadez\anaconda3\envs\surprise_env\python.exe" app.py
) else (
  python app.py
)

pause
