import os

# Only set local windows cache if on windows and not provided by environment
if os.name == "nt" and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "d:\\sih26167\\.huggingface_cache"
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.concurrency import run_in_threadpool
from typing import List, Optional
import uvicorn
import time
from collections import defaultdict
from agent_controller import agent_controller
import storage_manager
from cache_manager import cache_manager
from storage_r2 import r2_storage

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic Vision-Language System for Remote Sensing & Earth Observation (ISRO / SAC - PS 26167)",
    version="1.0.0"
)

# Enable GZip compression for responses (reduces GeoJSON & base64 payload transfer time)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Allow CORS for Next.js frontend (local and deployed on Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Rate Limiter & Security Middleware (Redis / In-Memory Sliding Window)
# -------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    # Exclude health check and preflight OPTIONS from rate limiting
    if request.url.path not in ["/", "/health"] and request.method != "OPTIONS":
        if not cache_manager.check_rate_limit(client_ip, limit=60, window_seconds=60):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Too Many Requests", "message": "Rate limit of 60 req/min exceeded. Please throttle requests."},
                headers={"Retry-After": "60"}
            )

    response = await call_next(request)
    return response

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".geotiff",
    ".webp", ".bmp", ".jp2", ".j2k", ".avif", ".fits", ".fit", ".gif",
    ".img", ".dat", ".bin", ""
}

START_TIME = time.time()

@app.get("/")
@app.get("/health")
async def healthcheck():
    """
    Comprehensive healthcheck endpoint for Render / Hugging Face Spaces / cloud detection.
    Reflects ISRO PS 26167 multi-model architecture specifications and free scaling status.
    """
    return {
        "status": "online",
        "service": "SatQuery AI Backend",
        "problem_statement": "ISRO / SAC — PS 26167",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "scaling": {
            "tier": "100% Free Production Stack ($0/month)",
            "cache_backend": cache_manager.get_backend_name(),
            "cache_diagnostics": cache_manager.get_status(),
            "storage_backend": storage_manager.get_storage_type(),
            "object_storage": "cloudflare_r2" if r2_storage.is_configured() else "inline_base64_fallback",
            "onnx_acceleration": "SatSegNet INT8 Quantized (<4ms CPU)"
        },
        "models": {
            "vlm_foundation": "Florence-2-base (Fine-tuned for Earth Observation / Remote Sensing)",
            "segmentation": "SatSegNet (ResNet-18 Backbone, 6 Land-Cover Classes, ONNX INT8)",
            "classes": ["Background", "Water", "Forest/Vegetation", "Bare Soil", "Urban/Built-up", "Agriculture"],
            "change_detection": "Bi-Temporal Residual Engine with Spatial Coherence Kernel",
            "sar_fusion": "RISAT-1 / Sentinel-1 C-Band Cloud-Penetrating Synthesis",
            "spatial_grounding": "Connected-component subpixel contour delineation (RFC 7946 GeoJSON)"
        },
        "gis_integration": {
            "bhuvan_wms": "enabled (bhuvan-vec2.nrsc.gov.in)",
            "geojson_export": "enabled (EPSG:4326 / RFC 7946)"
        }
    }

@app.post("/api/query")
async def process_query(
    query: str = Form(...),
    history: str = Form(None),
    images: List[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(None)
):
    """
    Endpoint to process natural language query with multimodal images in ANY format.
    Includes deterministic query caching (0ms hits) and persistent database storage.
    """
    # Optional API key verification (if configured in env)
    required_key = os.getenv("SATQUERY_API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

    if not images or len(images) == 0:
        return {
            "status": "error",
            "message": "Please provide at least one satellite image."
        }

    # Read image contents
    image_bytes_list = []
    for image in images:
        content = await image.read()
        image_bytes_list.append(content)

    # 1. Check Deterministic Hash Cache (0ms response, 0 RAM/CPU cost)
    cache_key = cache_manager.compute_query_hash(query, image_bytes_list)
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        result_copy = dict(cached_result)
        result_copy["from_cache"] = True
        result_copy["cache_key"] = cache_key
        return result_copy
        
    import json
    history_list = []
    if history:
        try:
            history_list = json.loads(history)
        except Exception:
            pass
            
    # Execute query in a threadpool so CPU-heavy CV/NumPy operations do not block the event loop
    result = await run_in_threadpool(agent_controller.execute_query, query, image_bytes_list, history_list)
    
    # Save to persistent database (Supabase PostgreSQL or SQLite)
    if result.get("status") == "success":
        try:
            result_id = storage_manager.save_query_result(
                query_text=query,
                response=result,
                geojson_data=result.get("geojson_data")
            )
            result["id"] = result_id
            
            # Store in cache for future identical queries
            cache_manager.set(cache_key, result, ttl_seconds=86400)
        except Exception as store_err:
            print(f"[Storage Warning] Could not persist query to DB: {store_err}")

    return result


@app.get("/api/results")
async def get_recent_results(limit: int = 20):
    """
    Returns recent analysis runs with metadata, confidence, and timestamps.
    """
    try:
        return {
            "status": "success",
            "results": storage_manager.list_recent_queries(limit=limit)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{result_id}")
async def get_result_by_id(result_id: str):
    """
    Retrieves full persistent query result, GeoJSON data, and execution summary by ID.
    """
    result = storage_manager.get_query_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Query result '{result_id}' not found.")
    return {
        "status": "success",
        "result": result
    }


from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS

load_dotenv()

class AcquireRequest(BaseModel):
    bbox: list  # [min_lon, min_lat, max_lon, max_lat]
    start_date: str
    end_date: str
    dataset: str
    maxcc: int
    configuration: Optional[str] = "Default"
    layer: Optional[str] = "True color"

def get_evalscript(dataset: str, layer: str) -> str:
    # Handle Sentinel-1 Radar which only has VV/VH
    if dataset == "s1":
        return """
        //VERSION=3
        function setup() {
          return {
            input: ["VV", "VH", "dataMask"],
            output: { bands: 4 }
          };
        }
        function evaluatePixel(sample) {
          return [2.0 * sample.VV, 2.0 * sample.VH, 1.5 * sample.VV, sample.dataMask];
        }
        """

    # For Sentinel-2 and Landsat 8, mapping bands
    if dataset == "l8":
        b_red, b_green, b_blue = "B04", "B03", "B02"
        b_nir = "B05"
        b_swir1 = "B06"
        b_swir2 = "B07"
        b_narrow_nir = "B05" # approximate
    else:
        # Default Sentinel-2
        b_red, b_green, b_blue = "B04", "B03", "B02"
        b_nir = "B08"
        b_swir1 = "B11"
        b_swir2 = "B12"
        b_narrow_nir = "B8A"

    if layer == "NDVI":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_nir}", "{b_red}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            let ndvi = (sample.{b_nir} - sample.{b_red}) / (sample.{b_nir} + sample.{b_red} + 0.0001);
            if (ndvi < 0) return [0.1, 0.1, 0.4, sample.dataMask]; // Water
            if (ndvi < 0.2) return [0.6, 0.4, 0.2, sample.dataMask]; // Bare soil
            if (ndvi < 0.5) return [0.6, 0.8, 0.2, sample.dataMask]; // Sparse vegetation
            return [0.1, 0.6, 0.1, sample.dataMask]; // Dense vegetation
        }}
        """
    elif layer == "False color":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_nir}", "{b_red}", "{b_green}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            return [2.5 * sample.{b_nir}, 2.5 * sample.{b_red}, 2.5 * sample.{b_green}, sample.dataMask];
        }}
        """
    elif layer == "False color (urban)":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_swir2}", "{b_swir1}", "{b_red}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            return [2.5 * sample.{b_swir2}, 2.5 * sample.{b_swir1}, 2.5 * sample.{b_red}, sample.dataMask];
        }}
        """
    elif layer == "SWIR":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_swir2}", "{b_swir1 if dataset == 'l8' else b_narrow_nir}", "{b_red}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            return [2.5 * sample.{b_swir2}, 2.5 * sample.{b_swir1 if dataset == 'l8' else b_narrow_nir}, 2.5 * sample.{b_red}, sample.dataMask];
        }}
        """
    elif layer == "NDWI":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_green}", "{b_nir}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            let ndwi = (sample.{b_green} - sample.{b_nir}) / (sample.{b_green} + sample.{b_nir} + 0.0001);
            if (ndwi > 0.1) return [0, 0.8, 1.0, sample.dataMask];
            return [2.5 * sample.{b_red}, 2.5 * sample.{b_green}, 2.5 * sample.{b_blue}, sample.dataMask];
        }}
        """
    elif layer == "Moisture index" and dataset != "l8":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_narrow_nir}", "{b_swir1}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            let ndmi = (sample.{b_narrow_nir} - sample.{b_swir1}) / (sample.{b_narrow_nir} + sample.{b_swir1} + 0.0001);
            // Convert index to a color gradient from brown to blue
            let r = ndmi < 0 ? 0.8 + ndmi : 0.8 - ndmi;
            let g = ndmi < 0 ? 0.6 + ndmi : 0.6 + ndmi;
            let b = ndmi < 0 ? 0.2 : 0.2 + ndmi * 2;
            return [r, g, b, sample.dataMask];
        }}
        """
    elif layer == "Wildfires" and dataset != "l8":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_swir2}", "{b_swir1}", "{b_blue}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            let active = (sample.{b_swir2} > 0.5 && sample.{b_swir1} > 0.3) ? 1 : 0;
            if (active) return [1.0, 0.2, 0.0, sample.dataMask]; // bright red for active fires
            return [2.5 * sample.{b_swir2}, 2.5 * sample.{b_swir1}, 2.5 * sample.{b_blue}, sample.dataMask];
        }}
        """
    elif layer == "Scene classification map" and dataset != "l8":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["SCL", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            let scl = sample.SCL;
            if (scl == 3) return [0.4, 0.2, 0.0, sample.dataMask]; // Shadow
            if (scl == 4) return [0.1, 0.6, 0.1, sample.dataMask]; // Vegetation
            if (scl == 5) return [0.6, 0.6, 0.1, sample.dataMask]; // Bare soil
            if (scl == 6) return [0.1, 0.1, 0.8, sample.dataMask]; // Water
            if (scl == 8 || scl == 9 || scl == 10) return [0.9, 0.9, 0.9, sample.dataMask]; // Cloud
            if (scl == 11) return [0.3, 0.9, 0.9, sample.dataMask]; // Snow
            return [0.0, 0.0, 0.0, 1.0];
        }}
        """
    elif layer == "NDSI":
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_green}", "{b_swir1}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            let ndsi = (sample.{b_green} - sample.{b_swir1}) / (sample.{b_green} + sample.{b_swir1} + 0.0001);
            if (ndsi > 0.4) return [0.0, 0.8, 1.0, sample.dataMask]; // Snow highlight
            return [0.2, 0.2, 0.2, sample.dataMask]; // Dark background
        }}
        """
    else:
        # True color or Default
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{b_red}", "{b_green}", "{b_blue}", "dataMask"],
                output: {{ bands: 4 }}
            }};
        }}
        function evaluatePixel(sample) {{
            return [2.5 * sample.{b_red}, 2.5 * sample.{b_green}, 2.5 * sample.{b_blue}, sample.dataMask];
        }}
        """

@app.get("/api/test-sentinel-credentials")
async def test_sentinel_credentials():
    """
    Diagnostic endpoint to verify Sentinel Hub / CDSE credentials.
    """
    import requests
    client_id = os.getenv("SH_CLIENT_ID", "").strip().strip('"\'')
    client_secret = os.getenv("SH_CLIENT_SECRET", "").strip().strip('"\'')
    sh_base_url = os.getenv("SH_BASE_URL", "").strip().strip('"\'')
    sh_token_url = os.getenv("SH_TOKEN_URL", "").strip().strip('"\'')

    if not client_id or not client_secret:
        return {
            "status": "missing_credentials",
            "error": "SH_CLIENT_ID or SH_CLIENT_SECRET environment variable is not set."
        }

    is_cdse = client_id.startswith("sh-") or ("dataspace.copernicus.eu" in sh_base_url)
    token_url = sh_token_url or (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        if is_cdse else
        "https://services.sentinel-hub.com/oauth/token"
    )

    try:
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret
            },
            timeout=10
        )
        if resp.status_code == 200:
            return {
                "status": "success",
                "message": "Sentinel Hub credentials verified successfully!",
                "platform": "Copernicus Data Space Ecosystem (CDSE)" if is_cdse else "Sentinel Hub (Sinergise)",
                "client_id_preview": f"{client_id[:8]}...{client_id[-4:]}"
            }
        else:
            return {
                "status": "auth_failed",
                "status_code": resp.status_code,
                "platform": "Copernicus Data Space Ecosystem (CDSE)" if is_cdse else "Sentinel Hub (Sinergise)",
                "token_url": token_url,
                "response": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
                "client_id_preview": f"{client_id[:8]}...{client_id[-4:]}",
                "guidance": (
                    "Common CDSE issues: 1) Ensure 'Client will be used by a single-page application' was left UNTICKED when creating the client. "
                    "2) Make sure the flow is 'Client Credentials'. 3) Copy the generated client secret immediately from the popup (not your login password). "
                    "4) Check that no quotes or whitespace were added on Render."
                )
            }
    except Exception as err:
        return {
            "status": "connection_error",
            "error": str(err)
        }

@app.post("/api/acquire")
async def acquire_imagery(request: AcquireRequest):
    """
    Endpoint to acquire satellite imagery from Sentinel Hub.
    """
    import requests
    client_id = os.getenv("SH_CLIENT_ID", "").strip().strip('"\'')
    client_secret = os.getenv("SH_CLIENT_SECRET", "").strip().strip('"\'')
    
    config = SHConfig()
    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
    
    # Support both standard Sentinel Hub and Copernicus Data Space Ecosystem (CDSE)
    sh_base_url = os.getenv("SH_BASE_URL", "").strip().strip('"\'')
    sh_token_url = os.getenv("SH_TOKEN_URL", "").strip().strip('"\'')
    if not sh_base_url and client_id and client_id.startswith("sh-"):
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    elif sh_base_url:
        config.sh_base_url = sh_base_url
        if sh_token_url:
            config.sh_token_url = sh_token_url
            
    # Try Sentinel Hub API first; if credentials or scene lookup fail, seamlessly fall back to high-res global satellite imagery
    sentinel_error_details = None
    if client_id and client_secret:
        try:
            # Convert bbox array to BBox object
            bbox_obj = BBox(bbox=request.bbox, crs=CRS.WGS84)
            
            # Select standard Data Collection mapped to Sentinel Hub
            collection = DataCollection.SENTINEL2_L2A
            if request.dataset == "s1":
                collection = DataCollection.SENTINEL1_IW
            elif request.dataset == "l8":
                collection = DataCollection.LANDSAT8_L2
                
            # Get Evalscript based on requested dataset and layer
            evalscript = get_evalscript(request.dataset, getattr(request, 'layer', 'True color'))
                
            # Build Request
            sh_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=collection,
                        time_interval=(request.start_date, request.end_date),
                        mosaicking_order="mostRecent",
                        maxcc=float(request.maxcc) / 100.0
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.PNG)
                ],
                bbox=bbox_obj,
                size=[1024, 1024], # Request standard 1024x1024
                config=config
            )
            
            # Execute Request (returns a list of responses, we take the first)
            response_list = sh_request.get_data(decode_data=False)
            
            if response_list and len(response_list) > 0:
                raw_image_bytes = response_list[0].content
                return Response(
                    content=raw_image_bytes, 
                    media_type="image/png",
                    headers={"X-Acquisition-Source": "sentinel-hub"}
                )
        except Exception as sh_err:
            sentinel_error_details = str(sh_err)
            print(f"[Acquisition Notice] Sentinel Hub returned error ({sh_err}). Engaging high-resolution Earth observation fallback...")
    else:
        sentinel_error_details = "Credentials not configured"

    # High-Resolution Global Satellite Imagery Fallback for the requested Bounding Box
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        fallback_url = (
            f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export?"
            f"bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&size=1024,1024&imageSR=4326&format=png&f=image"
        )
        resp = requests.get(
            fallback_url,
            headers={"User-Agent": "SatQuery-AI/1.0 (Earth Observation Intelligence Engine)"},
            timeout=20
        )
        if resp.status_code == 200 and len(resp.content) > 3000:
            fallback_bytes = resp.content
            layer = getattr(request, "layer", "True color")
            if layer not in ["True color", "Default", "Highlight Optimized Natural Color"]:
                try:
                    from io import BytesIO
                    from PIL import Image
                    import numpy as np
                    
                    img = Image.open(BytesIO(fallback_bytes)).convert("RGB")
                    arr = np.array(img).astype(float)
                    
                    if layer == "NDVI":
                        R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                        vari = (G - R) / (G + R - B + 1.0)
                        vari = np.clip(vari, -1, 1)
                        mapped = np.zeros_like(arr)
                        mapped[..., 1] = np.clip((vari + 1) * 127.5, 0, 255) # Green
                        mapped[..., 0] = np.clip(255 - (vari + 1) * 127.5, 0, 255) # Red
                        img = Image.fromarray(mapped.astype(np.uint8))
                    elif layer == "False color":
                        R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                        mapped = np.stack([G * 1.5, R, B], axis=2)
                        mapped = np.clip(mapped, 0, 255)
                        img = Image.fromarray(mapped.astype(np.uint8))
                    elif layer == "NDWI":
                        R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                        water = (B - G) / (B + G + 1.0)
                        mapped = np.zeros_like(arr)
                        mapped[..., 2] = np.clip(water * 255 * 2, 0, 255) # Blue
                        mapped[..., 1] = np.clip(water * 255, 0, 255) # Cyan
                        img = Image.fromarray(mapped.astype(np.uint8))
                    elif layer in ["SWIR", "Wildfires"]:
                        R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                        mapped = np.stack([R * 1.5, G * 0.8, B * 0.5], axis=2)
                        mapped = np.clip(mapped, 0, 255)
                        img = Image.fromarray(mapped.astype(np.uint8))
                    
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    fallback_bytes = buf.getvalue()
                except Exception as sim_err:
                    print(f"Failed to simulate layer {layer}: {sim_err}")
                    
            return Response(
                content=fallback_bytes, 
                media_type="image/png",
                headers={
                    "X-Acquisition-Source": "esri-world-imagery-auto-fallback",
                    "X-Acquisition-Layer": layer
                }
            )
    except Exception as fallback_err:
        print(f"[Fallback Error] Satellite imagery export failed: {fallback_err}")

    # Fallback to USGS/NASA static composite or informative error
    detail_msg = "Unable to acquire satellite imagery for the selected area."
    if sentinel_error_details:
        detail_msg += f" Sentinel Hub notice: {sentinel_error_details[:180]}."
    raise HTTPException(status_code=500, detail=detail_msg)

@app.get("/api/sample-missions")
async def get_sample_missions():
    """
    Endpoint returning pre-packaged mission scenarios with sample satellite pairs and prompts.
    """
    from sample_missions import generate_sample_missions
    return {
        "status": "success",
        "missions": generate_sample_missions()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

