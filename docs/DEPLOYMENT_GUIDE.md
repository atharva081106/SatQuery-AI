# SatQuery AI - Free Cloud Deployment & Scaling Guide
### Deploy Frontend on Vercel + Backend on 100% Free Infrastructure ($0/Month)

This guide provides step-by-step instructions to deploy and scale the entire SatQuery AI platform using **100% permanently free cloud hosting tiers** with zero credit card billing required.

---

## 1. 100% Free Production Architecture Overview

| Component | Free Cloud Provider | Free Tier Allocation | Responsibility |
| :--- | :--- | :--- | :--- |
| **Frontend** | **Vercel** | 100 GB bandwidth/mo, unlimited requests | Next.js 15 UI, 3D Globe, Swipe Comparator, PDF Exporter |
| **Backend & ML Engine** | **Hugging Face Spaces (Docker)** | **2 vCPU, 16 GB RAM, 50 GB Disk** (Free Forever) | FastAPI, ONNX Runtime (`SatSegNet` INT8), OpenCV, Rasterio |
| **Alternative Backend** | **Railway / Render** | $5 free credit / Free 512MB Web Service | Alternative FastAPI ASGI runner |
| **Spatial Database** | **Supabase (PostgreSQL + PostGIS)** | 500 MB DB, 50k MAU | Stores query telemetry, GeoJSON polygons with spatial indexes |
| **Distributed Cache** | **Upstash Serverless Redis** | 10,000 commands / day | Caches identical query hashes (0ms hits) & sliding-window rate limit |
| **Object Storage** | **Cloudflare R2** | 10 GB storage, 0 egress fees | Stores heavy satellite rasters, masks, and heatmaps |
| **Keep-Alive Cron** | **GitHub Actions** | 2,000 minutes / month | Automated cron pings `/health` every 12 mins (0 cold starts) |

> **Graceful Fallbacks**: Every free cloud service is optional. If unconfigured, the backend automatically falls back to local SQLite, in-memory LRU caching, and inline base64 image encoding.

---

## 2. Part 1: Push Codebase to GitHub

1. Open PowerShell / Terminal in `d:\sih26167`:
   ```powershell
   git status
   git add .
   git commit -m "feat(scale): 100% free scale stack with ONNX quantization, Redis cache, and Supabase"
   ```

2. Link and push your repository to your GitHub account:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/satquery-ai.git
   git push -u origin main
   ```

---

## 3. Part 2: Deploy Backend to Hugging Face Spaces (16 GB Free RAM)

*Why Hugging Face Spaces?* Most free container tiers (Render, Koyeb) limit RAM to 512MB, which risks OOM errors on large satellite rasters. Hugging Face Spaces provides **16 GB RAM completely free forever**.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set configuration:
   - **Space name**: `satquery-backend`
   - **License**: `mit` or `apache-2.0`
   - **SDK**: Select **Docker** -> **Blank**.
   - **Space Hardware**: Free `2 vCPU · 16 GB RAM`.
3. In your local repository, add Hugging Face as a secondary git remote (or push using the HF web UI):
   ```powershell
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/satquery-backend
   git push hf main
   ```
4. Hugging Face will automatically build the `backend/Dockerfile` and expose port `7860`.
5. Your public backend URL will be:
   `https://YOUR_HF_USERNAME-satquery-backend.hf.space`

---

## 4. Part 3: (Alternative) Deploy Backend to Render / Railway

### Option A: Render (Free Web Service)
1. Go to [render.com](https://render.com/) and click **"New +"** -> **"Web Service"**.
2. Connect your `satquery-ai` repository.
3. Configure:
   - **Root Directory**: `backend`
   - **Runtime**: `Docker` (or `Python 3`)
   - **Instance Type**: `Free`
4. Click **"Create Web Service"**. Public URL: `https://satquery-backend.onrender.com`.

### Option B: Railway (Fastest Setup)
1. Go to [railway.app](https://railway.app/) and click **"New Project"** -> **"Deploy from GitHub repo"**.
2. Select your repository -> Go to **Settings** -> set **Root Directory** to `/backend`.
3. Under **Networking**, click **"Generate Domain"**. Public URL: `https://satquery-backend-production.up.railway.app`.

---

## 5. Part 4: Optional Free Add-Ons (Supabase, Upstash, Cloudflare R2)

To unlock multi-worker scaling, add these free environment variables in your backend dashboard (Hugging Face Spaces Settings / Render Environment / Railway Variables):

### A. Cloud PostgreSQL Database (Neon or Supabase)
SatQuery AI supports any standard PostgreSQL connection URI:

**Option 1: Neon Serverless PostgreSQL (Recommended Free Tier)**
1. Create a free project at [neon.tech](https://neon.tech/).
2. Copy your connection string from the Neon Console.
3. Set backend environment variable on Render / local `.env`:
   ```env
   DATABASE_URL=postgresql://[USER]:[PASSWORD]@[ENDPOINT].neon.tech/[DBNAME]?sslmode=require
   ```

**Option 2: Supabase PostgreSQL + PostGIS**
```env
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres
```

### B. Redis Distributed Cache (Native Redis & Upstash)
SatQuery AI supports both native Redis TCP connections and Upstash Serverless Redis REST API:

**Option 1: Native Redis (Docker, Redis Cloud, AWS ElastiCache, or Railway Redis)**
```env
REDIS_URL=redis://default:[PASSWORD]@[HOST]:[PORT]
# Or TLS/SSL:
REDIS_URL=rediss://default:[PASSWORD]@[HOST]:[PORT]
```

**Option 2: Upstash Serverless Redis (Free 10k cmds/day)**
1. Create a free database at [upstash.com](https://upstash.com/).
2. Copy the REST URL and Token from the Upstash console.
3. Set backend environment variables:
   ```env
   UPSTASH_REDIS_REST_URL=https://[YOUR_DB].upstash.io
   UPSTASH_REDIS_REST_TOKEN=[YOUR_UPSTASH_TOKEN]
   ```

### C. Cloudflare R2 (Free Zero-Egress Storage)
1. Create an R2 bucket named `satquery-artifacts` at [dash.cloudflare.com](https://dash.cloudflare.com/).
2. Create an API token with read/write permissions.
3. Set backend environment variables:
   ```env
   R2_ACCOUNT_ID=[YOUR_ACCOUNT_ID]
   R2_ACCESS_KEY_ID=[YOUR_ACCESS_KEY]
   R2_SECRET_ACCESS_KEY=[YOUR_SECRET_KEY]
   R2_BUCKET_NAME=satquery-artifacts
   R2_PUBLIC_URL=https://pub-[HASH].r2.dev
   ```

---

## 6. Part 5: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com/) and click **"Add New..."** -> **"Project"**.
2. Import your `satquery-ai` repository.
3. Configure:
   - **Framework Preset**: `Next.js` (automatically detected).
   - **Root Directory**: Click **Edit** and select `frontend`.
4. Add Environment Variable:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your backend URL (e.g., `https://YOUR_HF_USERNAME-satquery-backend.hf.space` or `https://satquery-backend-production.up.railway.app`) without trailing slash.
5. Click **"Deploy"**!
6. Vercel generates your live URL in ~60 seconds (e.g., `https://satquery-ai.vercel.app`).

---

## 7. Part 6: Zero Cold Starts (Automated Keep-Alive)

Free cloud containers sleep after 15 minutes of inactivity. SatQuery AI includes an automated GitHub Actions keep-alive workflow ([.github/workflows/keep_alive.yml](file:///d:/sih26167/.github/workflows/keep_alive.yml)):

1. In your GitHub repository, go to **Settings** -> **Secrets and variables** -> **Actions**.
2. Click **"New repository secret"**:
   - **Name**: `SATQUERY_BACKEND_URL`
   - **Value**: Your live backend URL (e.g., `https://YOUR_HF_USERNAME-satquery-backend.hf.space`).
3. GitHub Actions will ping `/health` every 12 minutes completely free, guaranteeing **zero cold starts** during hackathon judging and live presentations!

---

## 8. Verification & Health Audit

Once deployed, visit `https://YOUR_BACKEND_URL/health`. It will confirm your active scale stack:
```json
{
  "status": "online",
  "scaling": {
    "tier": "100% Free Production Stack ($0/month)",
    "cache_backend": "in_memory_lru",
    "storage_backend": "sqlite",
    "object_storage": "inline_base64_fallback",
    "onnx_acceleration": "SatSegNet INT8 Quantized (<4ms CPU)"
  }
}
```
If you add Supabase or Upstash credentials, `storage_backend` and `cache_backend` automatically switch to `postgresql` and `upstash_rest` respectively!
