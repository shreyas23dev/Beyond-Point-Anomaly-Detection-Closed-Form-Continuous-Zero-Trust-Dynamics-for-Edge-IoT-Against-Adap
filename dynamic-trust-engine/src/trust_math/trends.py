"""Trend detection."""
from typing import Literal

def compute_trend(current_trust: float, previous_trust: float) -> Literal["RECOVERING", "DECLINING", "STABLE"]:
    """Determine trust trend based on successive values."""
    if current_trust > previous_trust:
        return "RECOVERING"
    elif current_trust < previous_trust:
        return "DECLINING"
    else:
        return "STABLE"
