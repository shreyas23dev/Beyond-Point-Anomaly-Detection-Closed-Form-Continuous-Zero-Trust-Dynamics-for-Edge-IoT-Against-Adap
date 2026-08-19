"""Validates policy configuration for correctness and conflicts."""
import logging
from src.policy_loader import PolicyConfig

logger = logging.getLogger(__name__)

class PolicyValidationError(Exception):
    """Raised when policies are invalid."""
    pass

class PolicyValidator:
    """Validates the loaded policy configuration before the engine starts."""
    
    VALID_ACTIONS = {"ALLOW", "VERIFY", "BLOCK"}
    
    @classmethod
    def validate(cls, config: PolicyConfig):
        """Run all validation checks."""
        cls._check_empty(config)
        cls._check_duplicates(config)
        cls._check_actions(config)
        cls._check_conditions(config)
        logger.info(f"Successfully validated {len(config.policies)} policies (Version {config.metadata.policy_version}).")

    @classmethod
    def _check_empty(cls, config: PolicyConfig):
        if not config.policies:
            raise PolicyValidationError("No policies defined.")

    @classmethod
    def _check_duplicates(cls, config: PolicyConfig):
        seen_ids = set()
        for p in config.policies:
            if p.policy_id in seen_ids:
                raise PolicyValidationError(f"Duplicate Policy ID found: {p.policy_id}")
            seen_ids.add(p.policy_id)

    @classmethod
    def _check_actions(cls, config: PolicyConfig):
        for p in config.policies:
            if p.action not in cls.VALID_ACTIONS:
                raise PolicyValidationError(
                    f"Unknown action '{p.action}' in policy '{p.policy_id}'. "
                    f"Must be one of {cls.VALID_ACTIONS}"
                )

    @classmethod
    def _check_conditions(cls, config: PolicyConfig):
        """Perform basic validation on the condition string syntax."""
        for p in config.policies:
            if not p.condition or not isinstance(p.condition, str):
                raise PolicyValidationError(f"Malformed condition in policy '{p.policy_id}'.")
            
            # Since conditions are evaluated using python `eval` with a safe context,
            # we just ensure it's not empty and doesn't contain dangerous builtins.
            dangerous_keywords = ["__import__", "exec", "eval", "open", "os."]
            if any(k in p.condition for k in dangerous_keywords):
                raise PolicyValidationError(f"Dangerous keyword found in condition for policy '{p.policy_id}'.")
