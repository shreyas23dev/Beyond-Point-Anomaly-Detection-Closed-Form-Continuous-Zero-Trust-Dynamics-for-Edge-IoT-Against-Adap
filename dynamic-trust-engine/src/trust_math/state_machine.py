"""Trust state classification and state machine."""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

State = Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]

def classify_state(trust_score: float, boundaries: dict[str, float]) -> State:
    """Classify trust score into a state based on configured boundaries."""
    if trust_score >= boundaries.get("HIGH", 0.80):
        return "HIGH"
    elif trust_score >= boundaries.get("MEDIUM", 0.60):
        return "MEDIUM"
    elif trust_score >= boundaries.get("LOW", 0.40):
        return "LOW"
    else:
        return "CRITICAL"

def log_state_transition(device_id: str, old_state: State, new_state: State, trust_score: float, anomaly_score: float):
    """Log state transitions."""
    if old_state != new_state:
        logger.info(
            f"STATE TRANSITION | Device: {device_id} | "
            f"Old State: {old_state} -> New State: {new_state} | "
            f"Trust: {trust_score:.4f} | Anomaly: {anomaly_score:.4f}"
        )
