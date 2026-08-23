import os
import json
import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import InferenceResult, HealthResponse
from .database import init_db, add_history, get_history, get_history_item
from .inference_service import inference_service

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB limit
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB
    init_db()
    # Create uploads dir if it doesn't exist
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Load model
    inference_service.load_resources()
    yield
    # Cleanup resources (if any)
    pass

app = FastAPI(title="CXRMate-2 Local API", lifespan=lifespan)

# Allow CORS for local dev if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_upload(file: UploadFile) -> Path:
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension {ext}")
    
    # Check size (reading first and then resetting cursor isn't always reliable, 
    # but we can enforce it as we save)
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = UPLOADS_DIR / unique_filename
    
    # Write in chunks and enforce size
    size = 0
    with open(save_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                f.close()
                save_path.unlink()
                raise HTTPException(status_code=400, detail="File too large (max 20MB)")
            f.write(chunk)
    return save_path

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model="aehrc/cxrmate-2",
        device=str(inference_service.device),
        model_loaded=inference_service.is_loaded
    )

@app.post("/api/infer", response_model=InferenceResult)
async def infer_single(file: UploadFile = File(...)):
    image_path = validate_upload(file)
    
    try:
        result = await inference_service.analyze_image(image_path)
        
        # Save to DB
        add_history(
            image_name=file.filename,
            inference_time_seconds=result.get("inference_time_seconds", 0.0),
            findings=result.get("findings", ""),
            impression=result.get("impression", ""),
            status=result.get("status", "failed"),
            error=result.get("error", "")
        )
        
        # Clean up temporary file
        try:
            image_path.unlink()
        except OSError:
            pass
            
        if result["status"] == "failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
            
        return InferenceResult(**result)
    except Exception as e:
        # Clean up on error
        if image_path.exists():
            image_path.unlink()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/infer/batch")
async def infer_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            image_path = validate_upload(file)
            result = await inference_service.analyze_image(image_path)
            
            add_history(
                image_name=file.filename,
                inference_time_seconds=result.get("inference_time_seconds", 0.0),
                findings=result.get("findings", ""),
                impression=result.get("impression", ""),
                status=result.get("status", "failed"),
                error=result.get("error", "")
            )
            
            try:
                image_path.unlink()
            except:
                pass
                
            results.append(result)
        except Exception as e:
            results.append({
                "image": file.filename,
                "status": "failed",
                "error": str(e)
            })
            
    return {"results": results}

@app.get("/api/history")
async def get_history_api():
    return get_history()

@app.get("/api/report/{item_id}/json")
async def download_json(item_id: int):
    item = get_history_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
        
    result = {
        "image": item["image_name"],
        "model": "aehrc/cxrmate-2",
        "device": "cpu",
        "inference_time_seconds": item["inference_time_seconds"],
        "findings": item["findings"],
        "impression": item["impression"],
        "timestamp": item["timestamp"],
        "status": item["status"]
    }
    return JSONResponse(
        content=result,
        headers={"Content-Disposition": f"attachment; filename=report_{item_id}.json"}
    )

@app.get("/api/report/{item_id}/txt")
async def download_txt(item_id: int):
    item = get_history_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
        
    content = f"""CXRMate-2 Report
================
Image: {item['image_name']}
Date: {item['timestamp']}
Inference Time: {item['inference_time_seconds']:.2f}s
Model: aehrc/cxrmate-2
Status: {item['status']}

FINDINGS
--------
{item['findings']}

IMPRESSION
----------
{item['impression']}

================
Research/educational use only. This AI-generated report is not a medical diagnosis and must be reviewed by a qualified radiologist or clinician.
"""
    return PlainTextResponse(
        content=content,
        headers={"Content-Disposition": f"attachment; filename=report_{item_id}.txt"}
    )

# Serve the frontend statically
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
