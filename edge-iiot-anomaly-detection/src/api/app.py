"""FastAPI application initialization."""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import config
from src.api.middleware import TimingMiddleware

logger = logging.getLogger(__name__)

# Global state for model and pipeline
ml_components = {
    "model": None,
    "pipeline": None,
    "metadata": None,
    "shap_explainer": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup."""
    logger.info("Starting up API, loading ML components...")
    
    model_path = config.paths.artifacts_models / "model.pkl"
    pipeline_path = config.paths.artifacts_models / "preprocessing_pipeline.pkl"
    metadata_path = config.paths.artifacts_metadata / "metadata.json"
    
    if not (model_path.exists() and pipeline_path.exists()):
        logger.error("Model artifacts not found! Run training first.")
        # We don't raise here so the app can start and return 503s or show docs
    else:
        try:
            from src.model.isolation_forest import AnomalyDetector
            from src.preprocessing.pipeline import PreprocessingPipeline
            from src.explainability.shap_explainer import TreeShapExplainer
            
            ml_components["model"] = AnomalyDetector.load(model_path)
            ml_components["pipeline"] = PreprocessingPipeline.load(pipeline_path)
            
            # Setup SHAP Explainer
            ml_components["shap_explainer"] = TreeShapExplainer(
                ml_components["model"].model, 
                config.paths.artifacts_plots
            )
            
            logger.info("Successfully loaded model, pipeline, and SHAP explainer.")
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    ml_components["metadata"] = json.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load ML components: {e}", exc_info=True)
            
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API...")
    ml_components.clear()


app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes here to avoid circular imports
from src.api.routes import router
app.include_router(router)
