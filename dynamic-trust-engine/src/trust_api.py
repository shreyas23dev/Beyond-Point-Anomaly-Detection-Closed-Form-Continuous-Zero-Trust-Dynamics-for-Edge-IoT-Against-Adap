"""FastAPI service for the Dynamic Trust Engine."""
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException

from src.trust_engine import DynamicTrustEngine
from src.trust_models import TrustRequest, TrustResponse, TrustExplanation

import os

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/trust_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Trust Engine",
    description="Mathematical trust computation engine consuming ML anomaly scores.",
    version="1.0.0"
)

# Global engine instance
engine = DynamicTrustEngine(
    config_path="config/trust_config.yaml",
    storage_path="data/trust_history.json"
)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "schema_version": "1.0.0"
    }

@app.get("/device/{device_id}")
def get_device_trust(device_id: str):
    """Get the current trust state of a specific device."""
    state = engine.storage.load_device(device_id)
    if not state:
        raise HTTPException(status_code=404, detail="Device not found")
    return state.model_dump()

@app.post("/trust", response_model=TrustResponse)
def update_trust(request: TrustRequest):
    """Process a new ML output and update trust."""
    try:
        response, explanation = engine.process_request(request)
        
        # Optionally log the rich explanation to a separate log or file here
        logger.info(f"Trust update completed. Explanation: {explanation.model_dump_json()}")
        
        return response
    except Exception as e:
        logger.error(f"Failed to process trust update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
