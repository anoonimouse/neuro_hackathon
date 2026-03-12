@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Installing Python dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  exit /b 1
)

if /I "%1"=="calibrate" (
  echo [2/3] Running one-time camera calibration...
  python vision_pipeline.py --mode calibrate
  if errorlevel 1 (
    echo Calibration failed.
    exit /b 1
  )
) else (
  echo [2/3] Skipping calibration.
)

echo [3/3] Launching Streamlit dashboard...
python -m streamlit run dashboard.py

endlocal
