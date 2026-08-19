"""API endpoints."""
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

from config.settings import config
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()

def get_ml_components():
    """Helper to get ML components from app state."""
    from src.api.app import ml_components
    if ml_components["model"] is None or ml_components["pipeline"] is None:
        raise HTTPException(status_code=503, detail="ML Model is not loaded or still initializing.")
    return ml_components

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        from src.api.app import ml_components
        is_ready = ml_components["model"] is not None
        version = ml_components["metadata"].get("model_version", "unknown") if ml_components.get("metadata") else "unknown"
        
        return HealthResponse(
            status="ready" if is_ready else "initializing",
            model_version=version,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
async def model_info():
    """Get model metadata and expected features."""
    components = get_ml_components()
    return components.get("metadata", {"message": "Metadata not available"})

def process_prediction(df: pd.DataFrame, components: dict) -> list[PredictionResponse]:
    """Process a dataframe through pipeline and model."""
    pipeline = components["pipeline"]
    model = components["model"]
    explainer = components["shap_explainer"]
    
    # 1. Preprocess
    try:
        X_processed = pipeline.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
        
    # 2. Predict
    try:
        results = model.predict_proba(X_processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    # 3. Explain (SHAP)
    try:
        if explainer:
            feature_names = pipeline.get_feature_names()
            shap_values = explainer.compute_shap_values(X_processed)
            
            for i, result in enumerate(results):
                result["shap_explanation"] = explainer.generate_local_explanation(
                    shap_values, feature_names, sample_idx=i
                )
    except Exception as e:
        # Don't fail the prediction if SHAP fails, just log it
        import logging
        logging.getLogger(__name__).warning(f"SHAP explanation failed: {e}")
        
    # 4. Format response
    responses = []
    for res in results:
        responses.append(PredictionResponse(
            raw_anomaly_score=res["raw_anomaly_score"],
            anomaly_threshold=res["anomaly_threshold"],
            is_anomaly=res["is_anomaly"],
            confidence=res["confidence"],
            shap_explanation=res.get("shap_explanation", {})
        ))
        
    return responses

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    components = get_ml_components()
    
    # Convert to DataFrame
    df = pd.DataFrame([request.features])
    
    results = process_prediction(df, components)
    return results[0]

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    components = get_ml_components()
    
    if len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")
        
    if len(request.samples) > config.api.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(request.samples)} exceeds maximum {config.api.max_batch_size}"
        )
        
    # Convert to DataFrame
    df = pd.DataFrame(request.samples)
    
    results = process_prediction(df, components)
    return BatchPredictionResponse(predictions=results)
