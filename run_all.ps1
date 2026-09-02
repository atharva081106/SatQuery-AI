Write-Host "======================================" -ForegroundColor Cyan
Write-Host " SatQuery AI - Automated End-to-End Execution" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 1. Stop existing processes if any (optional, keeping it simple for now)

# 2. Setup Data and Train ML Model
Write-Host "`n[1/4] Generating Dummy Data and Training VLM for 1 Epoch..." -ForegroundColor Yellow
cd d:\sih26167\ml_pipeline
& "d:\sih26167\backend\venv\Scripts\activate.ps1"

# Ensure ML requirements are installed in the venv
pip install -r requirements.txt accelerate>=1.1.0

# Set HuggingFace cache to D: drive because C: drive is out of space!
$env:HF_HOME = "d:\sih26167\.huggingface_cache"

Write-Host "Running PyTorch Training Loop (This will download the BLIP model if not cached)..." -ForegroundColor Yellow
# Run for just 1 epoch on the 10-image dummy dataset
python generate_dummy_data.py
python train_vqa.py --data_dir d:\sih26167\ml_pipeline\dummy_dataset --output_dir d:\sih26167\ml_pipeline\satquery_finetuned_model --epochs 1 --batch_size 2

# 3. Evaluate the Model
Write-Host "`n[2/4] Evaluating the fine-tuned model..." -ForegroundColor Yellow
python evaluate.py --model_path d:\sih26167\ml_pipeline\satquery_finetuned_model --test_data d:\sih26167\ml_pipeline\isro_evaluation_set.json

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
