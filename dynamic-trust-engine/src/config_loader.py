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
        self.default_policy_name = self.config_data.get("default_policy", "Default")
        self.states = self.config_data.get("states", {})
        self.policies = self.config_data.get("policies", {})
        
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
            return {}
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
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
                "recovery_steps": 20,
                "ema_beta": 0.8,
                "history_window": 50
            })
        return policy
