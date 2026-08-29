"""Anomaly Exponential Moving Average (EMA) formulation (Eq. 8)."""

def compute_anomaly_ema(anomaly_score: float, previous_ema: float, beta: float = 0.40) -> float:
    """Compute recursive Exponential Moving Average of anomaly scores.
    
    E_t = beta * A_t + (1 - beta) * E_{t-1}
    """
    return float(beta * anomaly_score + (1.0 - beta) * previous_ema)
