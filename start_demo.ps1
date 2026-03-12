$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host "[1/3] Installing Python dependencies..."
python -m pip install -r requirements.txt

if ($args.Count -gt 0 -and $args[0].ToLower() -eq 'calibrate') {
    Write-Host "[2/3] Running one-time camera calibration..."
    python vision_pipeline.py --mode calibrate
} else {
    Write-Host "[2/3] Skipping calibration."
}

Write-Host "[3/3] Launching Streamlit dashboard..."
python -m streamlit run dashboard.py
