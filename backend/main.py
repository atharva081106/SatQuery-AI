import os
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

