"""Qualitative trust state classification (Eq. 11)."""
from typing import Literal

State = Literal["HIGH", "MEDIUM", "LOW", "CRITICAL"]

def classify_state(trust_score: float, boundaries: dict[str, float] = None) -> State:
    """Classify continuous trust metric T_t in [0, 1] into qualitative policy states."""
    high_bound = boundaries.get("HIGH", 0.80) if boundaries else 0.80
    med_bound = boundaries.get("MEDIUM", 0.60) if boundaries else 0.60
    low_bound = boundaries.get("LOW", 0.30) if boundaries else 0.30

    if trust_score >= high_bound:
        return "HIGH"       # ALLOW
    elif trust_score >= med_bound:
        return "MEDIUM"     # ALLOW + LOG
    elif trust_score >= low_bound:
        return "LOW"        # VERIFY (Challenge)
    else:
        return "CRITICAL"   # BLOCK (Quarantine)
