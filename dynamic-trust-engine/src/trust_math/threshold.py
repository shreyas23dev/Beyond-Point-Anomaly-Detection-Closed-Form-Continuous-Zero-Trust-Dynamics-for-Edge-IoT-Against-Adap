"""Trust threshold calculation."""

def compute_trust_threshold(anomaly_threshold: float) -> float:
    """Compute trust threshold automatically from ML anomaly threshold.
    
    T_threshold = 1 - A_threshold
    """
    return 1.0 - anomaly_threshold
