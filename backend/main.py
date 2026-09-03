import os

# Only set local windows cache if on windows and not provided by environment
if os.name == "nt" and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "d:\\sih26167\\.huggingface_cache"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from agent_controller import agent_controller

app = FastAPI(title="SatQuery AI Backend")

# Allow CORS for Next.js frontend (local and deployed on Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

@app.get("/")
async def healthcheck():
    """Healthcheck endpoint for Render / cloud port detection"""
    return {"status": "online", "service": "SatQuery AI Backend"}

@app.post("/api/query")
async def process_query(
    query: str = Form(...),
    history: str = Form(None),
    images: List[UploadFile] = File(...)
):
    """
    Endpoint to process natural language query with multimodal images.
    """
    # Check input compatibility
    for image in images:
        ext = ""
        if image.filename:
            ext = "." + image.filename.rsplit(".", 1)[-1].lower() if "." in image.filename else ""
        
        if ext not in ALLOWED_EXTENSIONS:
            return {
                "status": "error",
                "message": f"Unsupported file format: {ext}. Only GeoTIFF, TIFF, PNG, and JPEG are supported."
            }

    # Read image contents
    image_bytes_list = []
    for image in images:
        content = await image.read()
        image_bytes_list.append(content)
        
    import json
    history_list = []
    if history:
        try:
            history_list = json.loads(history)
        except Exception:
            pass
            
    result = agent_controller.execute_query(query, image_bytes_list, history_list)
    return result

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

@app.post("/api/acquire")
async def acquire_imagery(request: AcquireRequest):
    """
    Endpoint to acquire satellite imagery from Sentinel Hub.
    """
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Sentinel Hub credentials not configured.")
        
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    
    # Use Copernicus Data Space Ecosystem (CDSE) endpoints
    # since Sentinel Hub transitioned to CDSE and user keys are registered there.
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    try:
        # Convert bbox array to BBox object
        bbox_obj = BBox(bbox=request.bbox, crs=CRS.WGS84)
        
        cdse_url = "https://sh.dataspace.copernicus.eu"
        
        # Select Data Collection mapped to CDSE
        collection = DataCollection.define(name="CDSE_S2", api_id="sentinel-2-l2a", service_url=cdse_url)
        if request.dataset == "s1":
            collection = DataCollection.define(name="CDSE_S1", api_id="sentinel-1-grd", service_url=cdse_url)
        elif request.dataset == "l8":
            collection = DataCollection.define(name="CDSE_L8", api_id="landsat-ot-l2", service_url=cdse_url)
            
        # Evalscript for True Color
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B03", "B02", "dataMask"],
                output: { bands: 4 }
            };
        }
        function evaluatePixel(sample) {
            return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02, sample.dataMask];
        }
        """
        
        # If Sentinel 1, evalscript needs to be different (VV/VH)
        if request.dataset == "s1":
            evalscript = """
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
        
        if not response_list or len(response_list) == 0:
            raise HTTPException(status_code=404, detail="No imagery found for the specified parameters.")
            
        raw_image_bytes = response_list[0]
        
        return Response(content=raw_image_bytes, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentinel Hub Error: {str(e)}")

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

