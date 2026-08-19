"""Evaluates inputs against loaded policies."""
import logging
from typing import Tuple
from src.policy_loader import PolicyConfig, PolicyDefinition
from src.zta_models import ZTAInput

logger = logging.getLogger(__name__)

class PolicyEngine:
    """Evaluates policies deterministically. O(N) over policies, effectively O(1) for small config."""
    
    # Priority resolution for equal priority rules
    ACTION_WEIGHTS = {"BLOCK": 3, "VERIFY": 2, "ALLOW": 1}

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.policies = config.policies
        self.policy_version = config.metadata.policy_version

    def evaluate(self, req: ZTAInput) -> Tuple[str, str, str, str]:
        """
        Evaluate rules against (T, T_th, S).
        Returns: (decision, reason, policy_id, evaluated_rule)
        """
        # Safe evaluation context
        context = {
            "trust_score": req.trust_score,
            "trust_threshold": req.trust_threshold,
            "trust_state": req.trust_state,
            "trend": req.trend,
            "device_id": req.device_id,
        }

        matches: list[PolicyDefinition] = []

        for p in self.policies:
            try:
                # Evaluate the python-like condition string
                if eval(p.condition, {"__builtins__": {}}, context):
                    matches.append(p)
            except Exception as e:
                logger.error(f"Error evaluating condition for policy {p.policy_id}: {e}")

        if not matches:
            raise ValueError(f"No matching policy found for device {req.device_id}. Evaluation failed.")

        # Priority Resolution
        # 1. Highest priority wins (lower number or higher number? Usually higher number = higher priority).
        # Let's assume higher integer = higher priority as per the plan.
        # 2. If equal priority: BLOCK > VERIFY > ALLOW.
        matches.sort(
            key=lambda p: (p.priority, self.ACTION_WEIGHTS.get(p.action, 0)),
            reverse=True
        )

        selected = matches[0]
        
        # Replace template variables in the reason string if any, though our static reasons don't strictly need it.
        return (selected.action, selected.reason, selected.policy_id, selected.condition)
