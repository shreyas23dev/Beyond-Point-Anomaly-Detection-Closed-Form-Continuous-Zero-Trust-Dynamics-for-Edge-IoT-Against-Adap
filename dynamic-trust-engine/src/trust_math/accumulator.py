"""Asymmetric Slow-Burn Evidence Accumulator (Eq. 9)."""

def compute_slow_burn_accumulator(
    anomaly_ema: float,
    previous_sb: float,
    gamma: float = 0.335,
    lambda_: float = 0.55,
    delta: float = 0.05,
    theta: float = 1.00
) -> tuple[float, bool]:
    """Compute asymmetric evidence accumulator score and trigger flag.
    
    SB_t = SB_{t-1} + lambda   if E_t > gamma
           max(0, SB_{t-1} - delta) otherwise
    I_t  = 1[SB_t > theta]
    """
    if anomaly_ema > gamma:
        new_sb = previous_sb + lambda_
    else:
        new_sb = max(0.0, previous_sb - delta)
        
    is_detected = bool(new_sb > theta)
    return float(new_sb), is_detected
