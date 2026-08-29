"""Configuration loader for YAML config."""
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class TrustConfig:
    """Loads and provides access to trust configuration."""
    
    def __init__(self, config_path: str | Path = "config/trust_config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
        self.default_policy_name = self.config_data.get("default_policy", "ESP32")
        self.states = self.config_data.get("states", {
            "HIGH": 0.80,
            "MEDIUM": 0.60,
            "LOW": 0.30,
            "CRITICAL": 0.00
        })
        self.slow_burn = self.config_data.get("slow_burn", {
            "ema_beta": 0.40,
            "accumulation_threshold_gamma": 0.335,
            "accumulation_step_lambda": 0.55,
            "decay_step_delta": 0.05,
            "trigger_threshold_theta": 1.00,
            "penalty_weight_p": 0.25
        })
        self.policies = self.config_data.get("policies", {})
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def get_policy(self, device_type: str) -> dict:
        """Get policy for a specific device type, or fallback to default."""
        policy = self.policies.get(device_type)
        if not policy:
            logger.warning(f"No policy found for device type '{device_type}'. Using '{self.default_policy_name}'.")
            policy = self.policies.get(self.default_policy_name, {
                "initial_trust": 1.0,
                "recovery_target": 0.90,
                "recovery_steps": 10,
                "history_window": 30
            })
        return policy
