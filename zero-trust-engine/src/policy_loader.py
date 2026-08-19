"""Policy definition schemas and loader."""
import logging
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PolicyDefinition(BaseModel):
    policy_id: str
    priority: int
    condition: str
    action: str
    reason: str
    description: str = ""

class PolicyMetadata(BaseModel):
    policy_version: str
    schema_version: str
    description: str = ""

class PolicyConfig(BaseModel):
    metadata: PolicyMetadata
    policies: list[PolicyDefinition]

class PolicyLoader:
    """Loads and parses policy.yaml into Pydantic models."""
    
    def __init__(self, filepath: str | Path = "config/policy.yaml"):
        self.filepath = Path(filepath)

    def load(self) -> PolicyConfig:
        """Load and parse the YAML file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Policy file not found at {self.filepath}")
            
        with open(self.filepath, "r") as f:
            data = yaml.safe_load(f)
            
        if not data:
            raise ValueError("Policy file is empty.")
            
        # Convert the dictionary of policies into a list
        policies_dict = data.get("policies", {})
        policies_list = []
        for pid, pdata in policies_dict.items():
            pdata["policy_id"] = pid
            policies_list.append(pdata)
            
        return PolicyConfig(
            metadata=PolicyMetadata(**data.get("metadata", {})),
            policies=[PolicyDefinition(**p) for p in policies_list]
        )
