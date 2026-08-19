import pytest
from src.zta_models import ZTAInput
from src.policy_loader import PolicyConfig, PolicyMetadata, PolicyDefinition
from src.policy_engine import PolicyEngine

@pytest.fixture
def test_config():
    return PolicyConfig(
        metadata=PolicyMetadata(policy_version="1.0", schema_version="1.0"),
        policies=[
            PolicyDefinition(
                policy_id="ALLOW_ALL",
                priority=10,
                condition="trust_score >= trust_threshold",
                action="ALLOW",
                reason="Allow"
            ),
            PolicyDefinition(
                policy_id="BLOCK_LOW",
                priority=20,
                condition="trust_state == 'LOW'",
                action="BLOCK",
                reason="Block"
            )
        ]
    )

def test_policy_engine(test_config):
    engine = PolicyEngine(test_config)
    
    # Should evaluate ALLOW
    req1 = ZTAInput(
        device_id="dev1",
        trust_score=0.9,
        trust_threshold=0.5,
        trust_state="HIGH",
        trend="STABLE",
        reason="Test",
        timestamp="2026-06-30T12:00:00Z"
    )
    decision, reason, pid, rule = engine.evaluate(req1)
    assert decision == "ALLOW"
    assert pid == "ALLOW_ALL"
    
    # Should evaluate BLOCK due to higher priority
    req2 = ZTAInput(
        device_id="dev2",
        trust_score=0.9, # Even if score is high
        trust_threshold=0.5,
        trust_state="LOW", # State forces block
        trend="STABLE",
        reason="Test",
        timestamp="2026-06-30T12:00:00Z"
    )
    decision, reason, pid, rule = engine.evaluate(req2)
    assert decision == "BLOCK"
    assert pid == "BLOCK_LOW"
