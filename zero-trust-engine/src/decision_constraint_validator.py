"""Validates inputs and outputs of the Decision Engine to guarantee determinism."""
import logging
from src.zta_models import ZTAInput, ZTAOutput

logger = logging.getLogger(__name__)

class ConstraintValidationError(ValueError):
    """Raised when a strict decision constraint fails."""
    pass

class DecisionConstraintValidator:
    """Enforces strict pre- and post-evaluation contracts."""

    @classmethod
    def validate_input(cls, req: ZTAInput):
        """Validate input before evaluation."""
        if req.trust_score < 0.0 or req.trust_score > 1.0:
            raise ConstraintValidationError("Trust score out of bounds [0,1]")
        if req.trust_threshold < 0.0 or req.trust_threshold > 1.0:
            raise ConstraintValidationError("Trust threshold out of bounds [0,1]")
        if not req.schema_version:
            raise ConstraintValidationError("Input missing schema_version")

    @classmethod
    def validate_output(cls, out: ZTAOutput):
        """Validate output before logging and returning."""
        if out.zta_decision not in {"ALLOW", "VERIFY", "BLOCK"}:
            raise ConstraintValidationError(f"Invalid decision generated: {out.zta_decision}")
        if not out.policy_id:
            raise ConstraintValidationError("Decision missing policy_id")
        if not out.audit_id:
            raise ConstraintValidationError("Decision missing audit_id")
        if not out.policy_version:
            raise ConstraintValidationError("Decision missing policy_version")
