"""Pydantic schemas for the Dynamic Trust Engine."""
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

class TrustRequest(BaseModel):
    """Input contract from the ML Engine."""
    device_id: str = Field(..., description="Unique identifier for the device")
    device_type: str = Field(default="ESP32", description="Device class (e.g., ESP32, ProcessControl, Gateway)")
    raw_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized anomaly score A_t in [0, 1]")
    anomaly_threshold: float = Field(default=0.4554, ge=0.0, le=1.0, description="Calibrated anomaly decision threshold A_thr")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="ML prediction confidence")
    timestamp: str = Field(..., description="ISO8601 timestamp of the telemetry observation")
    top_features: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional feature contributions")

class TrustResponse(BaseModel):
    """Output contract for Zero Trust Decision Point."""
    device_id: str
    timestamp: str
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Dynamic trust metric T_t in [0, 1]")
    anomaly_ema: float = Field(..., ge=0.0, le=1.0, description="Smoothed anomaly energy E_t in [0, 1]")
    slow_burn_score: float = Field(..., ge=0.0, description="Accumulated slow-burn evidence SB_t >= 0")
    slow_burn_detected: bool = Field(..., description="Slow-burn trigger indicator I_t = 1[SB_t > theta]")
    trust_threshold: float = Field(default=0.60, ge=0.0, le=1.0, description="Verification threshold")
    trust_state: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"] = Field(..., description="Qualitative Zero-Trust state")
    trend: Literal["RECOVERING", "DECLINING", "STABLE"] = Field(..., description="Short-term trust trajectory trend")
    formula_used: Literal["OVERT_DECAY", "BENIGN_RECOVERY", "SLOW_BURN_MITIGATION"] = Field(..., description="Governing transition rule ID")
    reason: str = Field(..., description="Human-readable justification string")
    schema_version: str = Field(default="1.0.0")

class TrustExplanation(BaseModel):
    """Rich two-level audit record for IEC 62443 compliance."""
    device_id: str
    timestamp: str
    formula_used: Literal["OVERT_DECAY", "BENIGN_RECOVERY", "SLOW_BURN_MITIGATION"]
    trust_before: float
    trust_after: float
    anomaly_score: float
    anomaly_ema: float
    slow_burn_score: float
    slow_burn_detected: bool
    anomaly_threshold: float
    trust_state_before: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]
    trust_state_after: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]
    trend: Literal["RECOVERING", "DECLINING", "STABLE"]
    reason: str
    top_features: Optional[List[Dict[str, Any]]] = None

class DeviceTrustState(BaseModel):
    """Internal persisted per-device state tuple (T_t, E_t, SB_t, I_t, c_t)."""
    device_id: str
    device_type: str
    trust_score: float = Field(default=1.0, ge=0.0, le=1.0)
    anomaly_ema: float = Field(default=0.0, ge=0.0, le=1.0)
    slow_burn_score: float = Field(default=0.0, ge=0.0)
    slow_burn_detected: bool = Field(default=False)
    trust_state: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"] = Field(default="HIGH")
    history: List[float] = Field(default_factory=list)
    last_update: str
    last_anomaly_score: float = Field(default=0.0)
    consecutive_clean_steps: int = Field(default=0)
