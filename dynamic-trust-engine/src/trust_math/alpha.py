"""Closed-Form Trust Recovery Coefficient Derivation (Theorem 1, Eq. 5)."""

def compute_alpha(target_trust: float = 0.90, recovery_steps: int = 10, initial_distrust: float = 0.0) -> float:
    """Compute analytical recovery coefficient alpha in closed form.
    
    alpha = 1 - (1 - T_target)^(1 / k)
    """
    if recovery_steps <= 0:
        raise ValueError("Recovery steps (k) must be a positive integer.")
    if not (0.0 < target_trust < 1.0):
        raise ValueError("Target trust (T_target) must lie strictly within (0, 1).")
        
    alpha = 1.0 - (1.0 - target_trust) ** (1.0 / recovery_steps)
    return float(alpha)
