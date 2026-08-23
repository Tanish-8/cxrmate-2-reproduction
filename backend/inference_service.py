import asyncio
import torch
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure we can import from the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cxrmate_inference import (
    load_processor,
    load_model,
    load_generation_config,
    run_inference
)

class InferenceService:
    def __init__(self):
        self.processor = None
        self.model = None
        self.generation_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = asyncio.Lock()
        self.is_loaded = False

    def load_resources(self):
        """Load the model and processor into memory. Run once at startup."""
        print("Initializing InferenceService...")
        self.processor = load_processor()
        self.model = load_model(self.device)
        self.generation_config = load_generation_config()
        self.is_loaded = True
        print("InferenceService initialized.")

    async def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Run inference sequentially. 
        Uses asyncio.Lock to prevent concurrent generation on CPU.
        """
        async with self.lock:
            # We must run the CPU-intensive PyTorch generation in a thread
            # so we don't block the asyncio event loop for other endpoints.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._run_inference_sync,
                image_path
            )
            return result
            
    def _run_inference_sync(self, image_path: Path) -> Dict[str, Any]:
        """Wrapper to call the existing run_inference function."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        
        return run_inference(
            image_path=image_path,
            processor=self.processor,
            model=self.model,
            generation_config=self.generation_config,
            device=self.device
        )

# Global singleton instance
inference_service = InferenceService()
