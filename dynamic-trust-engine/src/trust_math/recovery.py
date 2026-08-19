"""Trust recovery calculation."""

def compute_recovery(current_trust: float, alpha: float) -> float:
    """Compute trust recovery.
    
    T_(t+1) = T_t + alpha * (1 - T_t)
    """
    return current_trust + alpha * (1.0 - current_trust)
