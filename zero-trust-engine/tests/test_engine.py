import pytest
import sqlite3
from pathlib import Path
from src.database import DatabaseManager
from src.decision_logger import DecisionLogger
from src.zta_engine import ZeroTrustEngine
from src.policy_engine import PolicyEngine
from src.policy_loader import PolicyConfig, PolicyMetadata, PolicyDefinition
from src.zta_models import ZTAInput

@pytest.fixture
def db_manager(tmp_path):
    db_file = tmp_path / "test.db"
    return DatabaseManager(db_file)

@pytest.fixture
def zta_engine(db_manager):
    config = PolicyConfig(
        metadata=PolicyMetadata(policy_version="1.0", schema_version="1.0"),
        policies=[
            PolicyDefinition(
                policy_id="POL-001",
                priority=100,
                condition="trust_score >= trust_threshold",
                action="ALLOW",
                reason="Allow"
            )
        ]
    )
    policy_engine = PolicyEngine(config)
    logger = DecisionLogger(db_manager)
    return ZeroTrustEngine(policy_engine, logger)

def test_engine_integration(zta_engine, db_manager):
    req = ZTAInput(
        device_id="dev1",
        trust_score=0.8,
        trust_threshold=0.5,
        trust_state="HIGH",
        trend="STABLE",
        reason="Test",
        timestamp="2026-06-30T12:00:00Z"
    )
    
    out, expl = zta_engine.process_request(req)
    
    assert out.zta_decision == "ALLOW"
    assert out.policy_id == "POL-001"
    assert out.device_id == "dev1"
    assert out.audit_id is not None
    
    # Check SQLite
    rows = db_manager.fetch_all("SELECT * FROM policy_log")
    assert len(rows) == 1
    assert rows[0]['audit_id'] == out.audit_id
    assert rows[0]['decision'] == "ALLOW"
    assert rows[0]['trust_score'] == 0.8
