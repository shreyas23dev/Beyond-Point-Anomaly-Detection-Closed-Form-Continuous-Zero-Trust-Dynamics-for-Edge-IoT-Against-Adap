"""Exponential Moving Average (EMA) and slow-burn calculation."""

def compute_ema(current_trust: float, previous_ema: float, beta: float) -> float:
    """Compute Exponential Moving Average of trust.
    
    EMA_t = beta * T_t + (1 - beta) * EMA_(t-1)
    """
    return beta * current_trust + (1.0 - beta) * previous_ema
