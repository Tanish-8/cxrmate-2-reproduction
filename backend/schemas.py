from pydantic import BaseModel
from typing import List, Optional

class InferenceResult(BaseModel):
    image: str
    image_path: str
    status: str
    inference_time_seconds: float
    findings: str
    impression: str
    error: str = ""

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    model_loaded: bool

class HistoryItem(BaseModel):
    id: int
    image_name: str
    timestamp: str
    inference_time_seconds: float
    findings: str
    impression: str
    status: str
    error: str = ""
