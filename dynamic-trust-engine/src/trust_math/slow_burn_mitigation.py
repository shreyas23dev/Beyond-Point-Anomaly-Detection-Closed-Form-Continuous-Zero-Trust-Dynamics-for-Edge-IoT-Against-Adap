"""Progressive Slow-Burn Mitigation Penalty (Eq. 10)."""

def compute_slow_burn_mitigation(
    current_trust: float,
    anomaly_score: float,
    anomaly_threshold: float,
    penalty_p: float = 0.25
) -> float:
    """Compute progressive trust degradation under confirmed slow-burn evidence.
    
    T_{t+1} = max(0, T_t - P * (A_t / A_thr))
    """
    scaled_penalty = penalty_p * (anomaly_score / max(1e-6, anomaly_threshold))
    new_trust = max(0.0, current_trust - scaled_penalty)
    return float(min(1.0, new_trust))
