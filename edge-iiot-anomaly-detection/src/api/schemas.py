"""Pydantic schemas for the API."""
from pydantic import BaseModel, Field

from config.settings import config

class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: dict[str, float | int | str] = Field(
        ..., 
        description="Dictionary mapping feature names to their values."
    )

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    samples: list[dict[str, float | int | str]] = Field(
        ..., 
        description=f"List of feature dictionaries (max {config.api.max_batch_size}).",
        max_length=config.api.max_batch_size
    )

class PredictionResponse(BaseModel):
    """Prediction response matching the spec."""
    raw_anomaly_score: float = Field(..., description="Normalized isolation score (0-1), 1=most anomalous")
    anomaly_threshold: float = Field(..., description="Calibrated decision boundary")
    is_anomaly: bool = Field(..., description="True if score >= threshold")
    confidence: float = Field(..., description="Confidence score (0-1)")
    shap_explanation: dict[str, float] = Field(default_factory=dict, description="Per-feature SHAP contributions")
    schema_version: str = Field(default=config.schema_version, description="API schema version")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResponse]
    schema_version: str = Field(default=config.schema_version, description="API schema version")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_version: str
    timestamp: str
    schema_version: str = Field(default=config.schema_version)
