"""FastAPI application for Zero Trust Engine."""
import os
import json
import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.zta_models import ZTAInput, ZTAOutput
from src.database import DatabaseManager
from src.policy_loader import PolicyLoader
from src.policy_validator import PolicyValidator
from src.policy_engine import PolicyEngine
from src.decision_logger import DecisionLogger
from src.zta_engine import ZeroTrustEngine

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/zta_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global instances
db_manager = None
policy_config = None
zta_engine = None
start_time = time.time()

def generate_metadata(config):
    """Generates the zta_metadata.json artifact on startup."""
    metadata_path = Path("artifacts/zta_metadata.json")
    if not metadata_path.exists():
        metadata = {
            "engine": "Zero Trust Engine",
            "schema_version": config.metadata.schema_version,
            "policy_version": config.metadata.policy_version,
            "build": "2026.06",
            "author": "Fang Cybersecurity"
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Generated metadata at {metadata_path}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, policy_config, zta_engine, start_time
    start_time = time.time()
    
    # Initialize DB
    db_manager = DatabaseManager()
    
    # Load and validate policy
    loader = PolicyLoader()
    policy_config = loader.load()
    PolicyValidator.validate(policy_config)
    
    # Initialize Engine
    policy_engine = PolicyEngine(policy_config)
    decision_logger = DecisionLogger(db_manager)
    zta_engine = ZeroTrustEngine(policy_engine, decision_logger)
    
    generate_metadata(policy_config)
    logger.info("Zero Trust Engine started successfully.")
    
    yield
    
    logger.info("Shutting down Zero Trust Engine.")

app = FastAPI(
    title="Zero Trust Decision Engine",
    description="Evaluates trust inputs against configured policies to produce deterministic access decisions.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/decision", response_model=ZTAOutput)
def evaluate_decision(req: ZTAInput):
    """Main endpoint to evaluate ZTA decisions."""
    try:
        output, explanation = zta_engine.process_request(req)
        return output
    except Exception as e:
        logger.error(f"Decision evaluation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/policy")
def get_policy():
    """Returns the currently loaded policy configuration."""
    return policy_config.dict()

@app.get("/health")
def health_check():
    """Expanded health check endpoint."""
    uptime = int(time.time() - start_time)
    return {
        "status": "healthy",
        "database": "connected",
        "policy_loaded": policy_config is not None,
        "schema_version": policy_config.metadata.schema_version if policy_config else "unknown",
        "policy_version": policy_config.metadata.policy_version if policy_config else "unknown",
        "uptime_seconds": uptime
    }

@app.get("/stats")
def get_stats():
    """Returns aggregate decision statistics."""
    if db_manager:
        return db_manager.get_decision_stats()
    raise HTTPException(status_code=503, detail="Database not ready")
