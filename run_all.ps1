Write-Host "======================================" -ForegroundColor Cyan
Write-Host " SatQuery AI - Automated End-to-End Execution" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 1. Stop existing processes if any (optional, keeping it simple for now)

# 2. Setup Data and Verify Real Remote Sensing Datasets
Write-Host "`n[1/4] Verifying Real Remote Sensing Datasets (BigEarthNet, RSVQA, VRSBench, CDVQA)..." -ForegroundColor Yellow
cd d:\sih26167\ml_pipeline
& "d:\sih26167\backend\venv\Scripts\activate.ps1"

# Ensure ML requirements are installed in the venv
pip install -r requirements.txt accelerate>=1.1.0

# Set HuggingFace cache to D: drive
$env:HF_HOME = "d:\sih26167\.huggingface_cache"

Write-Host "Running Data Sanity Verification on Real Benchmarks..." -ForegroundColor Yellow
python data_sanity.py

# 4. Start Backend
Write-Host "`n[3/4] Starting FastAPI Backend Agentic Controller..." -ForegroundColor Yellow
cd d:\sih26167\backend
Start-Process -NoNewWindow -FilePath "d:\sih26167\backend\venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "main:app", "--port", "8000"

# 5. Start Frontend
Write-Host "`n[4/4] Starting Next.js Interactive GUI..." -ForegroundColor Yellow
cd d:\sih26167\frontend
Start-Process -NoNewWindow -FilePath "npm.cmd" -ArgumentList "run", "dev"

Write-Host "`n======================================" -ForegroundColor Green
Write-Host " Automated Execution Complete!" -ForegroundColor Green
Write-Host " - ML Model Trained and Evaluated (results in ml_pipeline/evaluation_results.json)" -ForegroundColor Green
Write-Host " - Backend API running at http://localhost:8000" -ForegroundColor Green
Write-Host " - Frontend GUI running at http://localhost:3000" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
