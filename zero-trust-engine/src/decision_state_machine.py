"""Enforces the strict lifecycle of a decision request."""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

State = Literal["REQUEST", "EVALUATING", "ALLOW", "VERIFY", "BLOCK", "AUDIT", "RETURNED"]

class DecisionStateMachine:
    """Tracks and enforces the state transitions of a single decision request."""
    
    VALID_TRANSITIONS = {
        "REQUEST": {"EVALUATING"},
        "EVALUATING": {"ALLOW", "VERIFY", "BLOCK"},
        "ALLOW": {"AUDIT"},
        "VERIFY": {"AUDIT"},
        "BLOCK": {"AUDIT"},
        "AUDIT": {"RETURNED"},
        "RETURNED": set()
    }

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.state: State = "REQUEST"

    def transition(self, new_state: State):
        """Transition to a new state."""
        if new_state not in self.VALID_TRANSITIONS.get(self.state, set()):
            raise ValueError(
                f"Invalid state transition for {self.device_id}: {self.state} -> {new_state}"
            )
        # We don't log every micro-step to avoid log spam, 
        # but the state machine enforces logical ordering internally.
        self.state = new_state
