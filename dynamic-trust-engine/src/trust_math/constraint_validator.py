"""Mathematical constraint validator."""
import math

class ConstraintViolationError(ValueError):
    """Raised when a mathematical constraint is violated."""
    pass

def validate_bounds(value: float, name: str, min_val: float = 0.0, max_val: float = 1.0):
    """Ensure value is within [min_val, max_val] and is finite."""
    if not math.isfinite(value):
        raise ConstraintViolationError(f"{name} must be finite, got {value}")
    if value < min_val or value > max_val:
        raise ConstraintViolationError(f"{name} bounds violated: {value} not in [{min_val}, {max_val}]")

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to bounds."""
    if not math.isfinite(value):
        return min_val # fallback for safety if something goes terribly wrong, though validate should catch it
    return max(min_val, min(max_val, value))

def validate_pre_update(raw_anomaly_score: float, anomaly_threshold: float, current_trust: float):
    """Validate inputs before performing a trust update."""
    validate_bounds(raw_anomaly_score, "raw_anomaly_score")
    validate_bounds(anomaly_threshold, "anomaly_threshold")
    validate_bounds(current_trust, "current_trust")

def validate_post_update(new_trust: float, alpha: float):
    """Validate outputs after a trust update."""
    validate_bounds(new_trust, "new_trust")
    validate_bounds(alpha, "alpha", min_val=0.0, max_val=1.0)
