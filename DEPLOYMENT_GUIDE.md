# SatQuery AI - Free Cloud Deployment Guide
### Deploy Frontend on Vercel + Backend on Railway or Render for Free

This guide provides step-by-step instructions to deploy the entire SatQuery AI platform using 100% free-tier cloud hosting.

---

## Architecture Overview
* **Frontend**: Next.js 15 deployed on **Vercel** (Free Hobby Tier).
* **Backend**: FastAPI + PyTorch + OpenCV deployed on **Railway** (Free $5 Credit) or **Render** (Free Web Service Tier).

---

## Part 1: Push Codebase to GitHub

Before deploying to Vercel or Railway/Render, ensure your code is in a GitHub repository:

1. Open PowerShell / Terminal in `d:\sih26167`:
   ```powershell
   git init
   git add .
   git commit -m "SatQuery AI cloud deployment ready"
   ```

2. Create a new repository on [GitHub](https://github.com/new) named `satquery-ai` (keep it Public or Private).

3. Link and push your repository:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/satquery-ai.git
   git push -u origin main
   ```
   *(Note: The pre-configured `.gitignore` ensures large `.huggingface_cache` weights and virtual environments are excluded so the upload completes in seconds).*

---

## Part 2: Deploy Backend to Railway (or Render)

### Option A: Deploying on Railway (Recommended for FastAPI + OpenCV)
1. Go to [railway.app](https://railway.app/) and sign in with GitHub.
2. Click **"+ New Project"** -> **"Deploy from GitHub repo"**.
3. Select your repository `satquery-ai`.
4. Click on the created service card -> go to **Settings**:
   - Set **Root Directory** to `/backend`.
   - Railway will automatically detect the `Dockerfile` or `Procfile`.
5. Under **Networking**, click **"Generate Domain"**.
   - You will receive a public HTTPS URL (e.g., `https://satquery-backend-production.up.railway.app`).
6. Copy this URL!

---

### Option B: Deploying on Render (Free Alternative)
1. Go to [render.com](https://render.com/) and sign in with GitHub.
2. Click **"New +"** -> **"Web Service"**.
3. Connect your `satquery-ai` repository.
4. Set the configuration:
   - **Name**: `satquery-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
5. Click **"Create Web Service"**.
6. Copy your public URL (e.g., `https://satquery-backend.onrender.com`).

---

## Part 3: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com/) and sign in with GitHub.
2. Click **"Add New..."** -> **"Project"**.
3. Import your `satquery-ai` repository.
4. In the configuration screen:
   - **Framework Preset**: `Next.js` (automatically detected).
   - **Root Directory**: Click **Edit** and choose `frontend`.
5. Expand **"Environment Variables"**:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: Paste your backend URL from Railway/Render (e.g. `https://satquery-backend-production.up.railway.app`) without trailing slash.
6. Click **"Deploy"**!
7. In ~60 seconds, Vercel will generate your live production URL (e.g. `https://satquery-ai.vercel.app`).

---

## Verification Checklist
1. Visit your Vercel URL in any browser or mobile device.
2. The landing page splash craft animation will play and load the mission HUD.
3. Open `/query`: upload a satellite image or tile pair.
4. The frontend will communicate over HTTPS with your cloud backend API, verify spatial compatibility, generate visual evidence, and output GeoJSON vectors!
