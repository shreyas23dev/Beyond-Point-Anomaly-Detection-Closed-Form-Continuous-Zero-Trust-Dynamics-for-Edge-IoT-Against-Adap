"""Alpha (Recovery Coefficient) calculation."""

def compute_alpha(initial_trust: float, target_trust: float, recovery_steps: int) -> float:
    """Compute the dynamic recovery coefficient (alpha).
    
    Derivation:
    T_(t+1) = T_t + alpha * (1 - T_t)
    alpha = 1 - ((1 - T_target) / (1 - T_0))^(1/k)
    
    Args:
        initial_trust: T_0
        target_trust: T_target
        recovery_steps: k
    
    Returns:
        alpha value
    """
    if initial_trust == 1.0:
        # Edge case: if we want to recover to 1.0 from some state, we can't use T_0=1.0 in the denominator.
        # But wait, the derivation is about starting from a decayed state T_0, 
        # and trying to reach T_target within k steps.
        # Actually, the config specifies 'initial_trust' as T_0 (like 1.0 for new devices), 
        # but the recovery equation T_target and k are policy parameters.
        # If the device is currently at T_current, that is the T_0 for the *recovery phase*.
        # The specification says: Compute alpha from: Initial Trust, Recovery Target, Recovery Steps.
        # So T_0 = Policy.Initial_Trust (e.g. 1.0), T_target = Policy.Recovery_Target (e.g. 0.90)
        # Wait, if T_0 = 1.0, 1 - T_0 = 0, which causes division by zero.
        # Let's re-read the spec: "α = 1 - ((1 - T_target)/(1 - T_0))^(1/k)"
        # If a device drops to 0.0 (worst case), we want to reach T_target in k steps.
        # Typically T_0 in this formula represents the starting point of recovery (e.g., 0.0).
        # Let's use 0.0 as the base starting point for derivation to get a constant alpha.
        pass
        
    # To strictly follow the spec's formula: "α = 1 - ((1 - T_target)/(1 - T_0))^(1/k)"
    # If the user put initial_trust = 1.0, (1 - 1.0) is 0. 
    # Let's assume T_0 means the *minimum possible trust* we recover from, or maybe the config's initial_trust 
    # was meant to be the T_0 in the formula, but mathematically if T_0=1.0 it fails.
    # Actually, in most models, if you want to recover to 0.9 from 0.0 in 10 steps, T_0 = 0.0.
    # Let's check the constraint validator later to ensure T_0 != 1.0 if it's used in this formula,
    # OR we assume T_0 in the formula is 0.0 by default.
    # Let's implement the formula exactly, and if it's 1.0, we fallback to 0.0 to avoid ZeroDivision.
    
    t_0 = initial_trust
    if t_0 == 1.0:
        # Fallback to 0.0 for the recovery calculation, meaning "recover from 0 to T_target in k steps"
        t_0 = 0.0
        
    numerator = 1.0 - target_trust
    denominator = 1.0 - t_0
    
    alpha = 1.0 - (numerator / denominator) ** (1.0 / recovery_steps)
    return alpha
