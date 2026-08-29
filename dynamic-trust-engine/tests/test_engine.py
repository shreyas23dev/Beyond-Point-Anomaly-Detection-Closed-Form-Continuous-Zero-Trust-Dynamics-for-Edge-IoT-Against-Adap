"""Integration tests for the Dynamic Trust Engine (Algorithm 1)."""
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
            config_path="config/trust_config.yaml",
            storage_path=storage_path
        )
        yield engine

def test_overt_attack_decay_branch(temp_engine):
    """Test Branch 1: Overt Attack Decay when A_t >= A_thr."""
    req = TrustRequest(
        device_id="dev_overt",
        device_type="ESP32",
        raw_anomaly_score=0.75,
        anomaly_threshold=0.4554,
        confidence=0.99,
        timestamp="2026-08-29T12:00:00Z"
    )
    
    resp, expl = temp_engine.process_request(req)
    
    assert resp.device_id == "dev_overt"
    assert resp.trust_score == pytest.approx(0.25)  # 1.0 * (1 - 0.75)
    assert resp.trust_state == "CRITICAL"           # 0.25 < 0.30
    assert resp.trend == "DECLINING"
    assert resp.formula_used == "OVERT_DECAY"
    assert expl.formula_used == "OVERT_DECAY"

def test_benign_recovery_branch(temp_engine):
    """Test Branch 2: Asymptotic Recovery when A_t < A_thr and I_t = 0."""
    # First: drop trust with moderate overt attack
    req1 = TrustRequest(
        device_id="dev_recov",
        device_type="ESP32",
        raw_anomaly_score=0.50,
        anomaly_threshold=0.4554,
        confidence=0.99,
        timestamp="2026-08-29T12:00:00Z"
    )
    temp_engine.process_request(req1)
    
    # Second: clean telemetry observation (A_t = 0.02)
    req2 = TrustRequest(
        device_id="dev_recov",
        device_type="ESP32",
        raw_anomaly_score=0.02,
        anomaly_threshold=0.4554,
        confidence=0.99,
        timestamp="2026-08-29T12:01:00Z"
    )
    resp, expl = temp_engine.process_request(req2)
    
    # Trust was 0.50. With alpha = 0.20567: T2 = 0.50 + 0.20567 * 0.50 = 0.6028
    assert resp.trust_score == pytest.approx(0.6028, abs=1e-3)
    assert resp.trust_state == "MEDIUM"
    assert resp.trend == "RECOVERING"
    assert resp.formula_used == "BENIGN_RECOVERY"

def test_subthreshold_slow_burn_mitigation_branch(temp_engine):
    """Test Branch 3: Sub-threshold Slow-Burn Detection and Mitigation on Real DDoS_HTTP Sequence."""
    # Real sub-threshold sample sequence with mean A_t = 0.4485 < A_thr = 0.4554
    subthreshold_scores = [0.445, 0.448, 0.450, 0.449, 0.452, 0.451, 0.450, 0.449]
    
    responses = []
    for i, score in enumerate(subthreshold_scores):
        req = TrustRequest(
            device_id="dev_slowburn",
            device_type="ESP32",
            raw_anomaly_score=score,
            anomaly_threshold=0.4554,
            confidence=0.95,
            timestamp=f"2026-08-29T12:0{i}:00Z"
        )
        resp, _ = temp_engine.process_request(req)
        responses.append(resp)
        
    # By step 4 (1-indexed step 5), accumulator exceeds theta = 1.0 -> slow_burn_detected is True
    assert any(r.slow_burn_detected for r in responses)
    # Formula used shifts to SLOW_BURN_MITIGATION
    assert any(r.formula_used == "SLOW_BURN_MITIGATION" for r in responses)
    # Trust score progressively degrades into LOW / CRITICAL
    final_resp = responses[-1]
    assert final_resp.trust_score < 0.60
    assert final_resp.trust_state in ["LOW", "CRITICAL"]

def test_benign_dormancy(temp_engine):
    """Test RQ4: Benign telemetry maintains high trust without triggering slow-burn."""
    clean_scores = [0.02, 0.05, 0.01, 0.03, 0.02, 0.04, 0.02, 0.01, 0.03, 0.02]
    
    for i, score in enumerate(clean_scores):
        req = TrustRequest(
            device_id="dev_clean",
            device_type="ESP32",
            raw_anomaly_score=score,
            anomaly_threshold=0.4554,
            confidence=0.99,
            timestamp=f"2026-08-29T13:0{i}:00Z"
        )
        resp, _ = temp_engine.process_request(req)
        assert not resp.slow_burn_detected
        assert resp.trust_score > 0.95
        assert resp.trust_state == "HIGH"
