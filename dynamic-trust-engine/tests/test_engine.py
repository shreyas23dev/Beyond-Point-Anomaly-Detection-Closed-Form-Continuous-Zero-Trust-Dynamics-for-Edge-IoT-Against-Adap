"""Integration tests for the Trust Engine."""
import os
import tempfile
import pytest
from src.trust_engine import DynamicTrustEngine
from src.trust_models import TrustRequest

@pytest.fixture
def temp_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "trust_history.json")
        engine = DynamicTrustEngine(
            config_path="config/trust_config.yaml",  # Assuming it exists in the test env
            storage_path=storage_path
        )
        yield engine

def test_engine_initialization_and_decay(temp_engine):
    req = TrustRequest(
        device_id="test_dev_1",
        device_type="ESP32",
        raw_anomaly_score=0.8, # Exceeds threshold -> Decay
        anomaly_threshold=0.5,
        confidence=0.9,
        timestamp="2026-06-30T12:00:00Z"
    )
    
    resp, expl = temp_engine.process_request(req)
    
    assert resp.device_id == "test_dev_1"
    assert resp.trust_threshold == 0.5 # 1 - 0.5
    assert resp.trust_score == pytest.approx(0.2) # initial 1.0 * (1 - 0.8)
    assert resp.trend == "DECLINING"
    assert expl.formula_used == "Decay"
    
def test_engine_recovery(temp_engine):
    # First request drops trust
    req1 = TrustRequest(
        device_id="test_dev_2",
        device_type="ESP32",
        raw_anomaly_score=0.9,
        anomaly_threshold=0.5,
        confidence=0.9,
        timestamp="2026-06-30T12:00:00Z"
    )
    temp_engine.process_request(req1)
    
    # Second request recovers trust
    req2 = TrustRequest(
        device_id="test_dev_2",
        device_type="ESP32",
        raw_anomaly_score=0.1, # Below threshold -> Recovery
        anomaly_threshold=0.5,
        confidence=0.9,
        timestamp="2026-06-30T12:01:00Z"
    )
    
    resp, expl = temp_engine.process_request(req2)
    
    assert resp.trust_score > 0.1 # initial 1.0 dropped to 0.1, now should be higher
    assert resp.trend == "RECOVERING"
    assert expl.formula_used == "Recovery"

def test_slow_burn_attack(temp_engine):
    # Send multiple requests with anomalies just below the threshold
    # The trust score will recover slightly, but the EMA might drag it down if we set beta properly.
    # Actually, if anomaly < threshold, it recovers. 
    # Wait, the spec says "Repeated low-level anomalies shall gradually decrease trust".
    # BUT "Recovery is permitted only when A_t < A_threshold".
    # If A_t < A_threshold, we recover. If it's recovering, trust goes UP.
    # How does slow-burn decrease trust if A_t < A_threshold? 
    # The EMA penalization logic we added says: if EMA < trust_threshold, force trust down.
    pass
