"""Pydantic schemas for the Zero Trust Decision Engine."""
from typing import Literal
from pydantic import BaseModel, Field

class ZTAInput(BaseModel):
    """Input contract from the Dynamic Trust Engine (Phase 2)."""
    device_id: str = Field(..., description="Unique identifier for the device")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Computed trust score")
    trust_threshold: float = Field(..., ge=0.0, le=1.0, description="Dynamic trust threshold")
    trust_state: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"] = Field(..., description="Categorical trust state")
    trend: Literal["RECOVERING", "DECLINING", "STABLE"] = Field(..., description="Trust trajectory")
    reason: str = Field(..., description="Reason for the current trust state")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    schema_version: str = Field(default="1.0.0", description="Schema version of the incoming payload")

class ZTAOutput(BaseModel):
    """Output contract for the End-to-End Integration (Phase 4)."""
    audit_id: str = Field(..., description="Unique UUID for this decision audit")
    device_id: str = Field(..., description="Unique identifier for the device")
    zta_decision: Literal["ALLOW", "VERIFY", "BLOCK"] = Field(..., description="Deterministic access decision")
    reason: str = Field(..., description="Explanation for the decision")
    policy_id: str = Field(..., description="The ID of the policy rule that was evaluated")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    schema_version: str = Field(default="1.0.0", description="Schema version of this payload")
    policy_version: str = Field(default="1.0.0", description="Version of the policy configuration used")

class DecisionExplanation(BaseModel):
    """Rich explainability payload for auditing and dashboards."""
    zta_decision: Literal["ALLOW", "VERIFY", "BLOCK"]
    reason: str
    policy_id: str
    evaluated_rule: str
    trust_score: float
    trust_threshold: float
    trust_state: Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]
