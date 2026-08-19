"""Main Zero Trust Decision Engine Orchestrator."""
import uuid
from datetime import datetime, timezone
import logging
from typing import Tuple

from src.zta_models import ZTAInput, ZTAOutput, DecisionExplanation
from src.decision_constraint_validator import DecisionConstraintValidator
from src.decision_state_machine import DecisionStateMachine
from src.policy_engine import PolicyEngine
from src.decision_logger import DecisionLogger

logger = logging.getLogger(__name__)

class ZeroTrustEngine:
    """Orchestrates policy evaluation deterministically."""

    def __init__(self, policy_engine: PolicyEngine, decision_logger: DecisionLogger):
        self.policy_engine = policy_engine
        self.decision_logger = decision_logger

    def process_request(self, req: ZTAInput) -> Tuple[ZTAOutput, DecisionExplanation]:
        """
        Main decision algorithm:
        1. Validate Constraints (Input)
        2. Evaluate Policies -> Decision
        3. Validate Constraints (Output)
        4. Log Decision
        """
        # Initialize state machine
        state_machine = DecisionStateMachine(req.device_id)

        # 1. Validate Input Constraints
        DecisionConstraintValidator.validate_input(req)

        # 2. Evaluate Policy
        state_machine.transition("EVALUATING")
        decision, reason, policy_id, evaluated_rule = self.policy_engine.evaluate(req)
        
        # Transition state machine based on decision
        if decision == "ALLOW":
            state_machine.transition("ALLOW")
        elif decision == "VERIFY":
            state_machine.transition("VERIFY")
        elif decision == "BLOCK":
            state_machine.transition("BLOCK")

        # 3. Generate Output & Audit ID
        audit_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        output = ZTAOutput(
            audit_id=audit_id,
            device_id=req.device_id,
            zta_decision=decision,
            reason=reason,
            policy_id=policy_id,
            timestamp=timestamp,
            schema_version="1.0.0",
            policy_version=self.policy_engine.policy_version
        )

        # Output Constraint Validation
        DecisionConstraintValidator.validate_output(output)

        # 4. Log Decision
        state_machine.transition("AUDIT")
        self.decision_logger.log_full_audit(
            out=output,
            req_trust_score=req.trust_score,
            req_trust_threshold=req.trust_threshold,
            req_trust_state=req.trust_state
        )

        state_machine.transition("RETURNED")
        
        # Explainability payload
        explanation = DecisionExplanation(
            zta_decision=decision,
            reason=reason,
            policy_id=policy_id,
            evaluated_rule=evaluated_rule,
            trust_score=req.trust_score,
            trust_threshold=req.trust_threshold,
            trust_state=req.trust_state
        )

        return output, explanation
