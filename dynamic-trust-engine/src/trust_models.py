"""Pydantic schemas for the Dynamic Trust Engine."""
from typing import Literal
from pydantic import BaseModel, Field

class TrustRequest(BaseModel):
    """Input contract from the ML Engine."""
    device_id: str = Field(..., description="Unique identifier for the device")
    device_type: str = Field(..., description="Device class (e.g., ESP32, PLC, Gateway)")
    raw_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized anomaly score (0-1)")
    anomaly_threshold: float = Field(..., ge=0.0, le=1.0, description="Learned anomaly threshold (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ML prediction confidence (0-1)")
    timestamp: str = Field(..., description="ISO8601 timestamp of the anomaly prediction")
    top_features: list[dict] | None = Field(default=None, description="Optional SHAP feature contributions")

class TrustResponse(BaseModel):
    """Output contract for Phase 3 (Zero Trust Decision Engine)."""
    device_id: str
    timestamp: str
    trust_score: float = Field(..., ge=0.0, le=1.0)
    trust_threshold: float = Field(..., ge=0.0, le=1.0)
    trust_state: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]
    trend: Literal["RECOVERING", "DECLINING", "STABLE"]
    reason: str
    ema: float = Field(..., ge=0.0, le=1.0)
    schema_version: str = Field(default="1.0.0")

class TrustExplanation(BaseModel):
    """Rich explainability output."""
    device_id: str
    timestamp: str
    formula_used: Literal["Decay", "Recovery"]
    trust_before: float
    trust_after: float
    anomaly_score: float
    anomaly_threshold: float
    trust_threshold: float
    trend: Literal["RECOVERING", "DECLINING", "STABLE"]
    reason: str
    top_features: list[dict] | None = None

class DeviceTrustState(BaseModel):
    """Internal persisted state for a device."""
    device_id: str
    device_type: str
    trust_score: float
    trust_state: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]
    ema: float
    history: list[float]
    last_update: str
    last_anomaly_score: float
