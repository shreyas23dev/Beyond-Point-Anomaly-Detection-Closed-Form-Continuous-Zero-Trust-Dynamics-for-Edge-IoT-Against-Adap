"""Trust decay calculation."""

def compute_decay(current_trust: float, anomaly_score: float) -> float:
    """Compute trust decay.
    
    T_(t+1) = T_t * (1 - A_t)
    """
    return current_trust * (1.0 - anomaly_score)
