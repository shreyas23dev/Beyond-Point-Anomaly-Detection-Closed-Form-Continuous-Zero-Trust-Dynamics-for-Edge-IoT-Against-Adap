"""Unit tests for Trust Engine mathematical formulations."""
import pytest
from src.trust_math.alpha import compute_alpha
from src.trust_math.decay import compute_decay
from src.trust_math.recovery import compute_recovery
from src.trust_math.ema import compute_anomaly_ema
from src.trust_math.accumulator import compute_slow_burn_accumulator
from src.trust_math.slow_burn_mitigation import compute_slow_burn_mitigation
from src.trust_math.state_machine import classify_state

def test_theorem1_alpha_derivation():
    """Verify Theorem 1: alpha = 1 - (1 - T_target)^(1/k)."""
    # ESP32: k=10, T_target=0.90 -> alpha approx 0.20567
    alpha_esp = compute_alpha(target_trust=0.90, recovery_steps=10)
    assert alpha_esp == pytest.approx(0.20567, abs=1e-4)
    
    # ProcessControl: k=30, T_target=0.90 -> alpha approx 0.07398
    alpha_plc = compute_alpha(target_trust=0.90, recovery_steps=30)
    assert alpha_plc == pytest.approx(0.07398, abs=1e-4)
    
    # Verify exact convergence in k steps from T0=0
    T = 0.0
    for _ in range(10):
        T = compute_recovery(T, alpha_esp)
    assert T == pytest.approx(0.90, abs=1e-5)

def test_multiplicative_decay():
    """Verify overt attack multiplicative decay: T_{t+1} = T_t * (1 - A_t)."""
    T0 = 1.0
    # Moderate attack A_t = 0.50 -> T1 = 0.50
    assert compute_decay(T0, 0.50) == pytest.approx(0.50)
    # Severe attack A_t = 0.95 -> T1 = 0.05
    assert compute_decay(T0, 0.95) == pytest.approx(0.05)
    # Consecutive attacks: 1.0 * 0.05 * 0.05 = 0.0025
    T2 = compute_decay(compute_decay(T0, 0.95), 0.95)
    assert T2 == pytest.approx(0.0025)

def test_asymmetric_accumulator_gain_and_decay():
    """Verify asymmetric accumulation (lambda=0.55) vs decay (delta=0.05)."""
    sb = 0.0
    # Step 1: E_t > gamma (0.35 > 0.335) -> +0.55
    sb, det = compute_slow_burn_accumulator(0.35, sb, gamma=0.335, lambda_=0.55, delta=0.05, theta=1.0)
    assert sb == pytest.approx(0.55)
    assert not det
    
    # Step 2: E_t > gamma -> +0.55 -> 1.10 (Trigger asserted: 1.10 > 1.0)
    sb, det = compute_slow_burn_accumulator(0.36, sb, gamma=0.335, lambda_=0.55, delta=0.05, theta=1.0)
    assert sb == pytest.approx(1.10)
    assert det
    
    # Step 3: E_t <= gamma (Clean step: 0.10 <= 0.335) -> -0.05 -> 1.05
    sb, det = compute_slow_burn_accumulator(0.10, sb, gamma=0.335, lambda_=0.55, delta=0.05, theta=1.0)
    assert sb == pytest.approx(1.05)
    assert det # Still > 1.0
    
    # Test strict non-negativity
    sb_zero = 0.02
    sb_zero, _ = compute_slow_burn_accumulator(0.10, sb_zero, gamma=0.335, lambda_=0.55, delta=0.05)
    assert sb_zero == 0.0

def test_zero_drift_anti_pacing_limit():
    """Verify theoretical zero-drift limit r* = delta / (lambda + delta) = 8.33%."""
    lambda_ = 0.55
    delta = 0.05
    r_star = delta / (lambda_ + delta)
    assert r_star == pytest.approx(0.08333, abs=1e-4)

def test_qualitative_state_classification():
    """Verify 4-tier Zero-Trust graduated state boundaries."""
    assert classify_state(0.95) == "HIGH"      # >= 0.80 -> ALLOW
    assert classify_state(0.80) == "HIGH"
    assert classify_state(0.75) == "MEDIUM"    # [0.60, 0.80) -> ALLOW + LOG
    assert classify_state(0.60) == "MEDIUM"
    assert classify_state(0.45) == "LOW"       # [0.30, 0.60) -> VERIFY (Step-up)
    assert classify_state(0.30) == "LOW"
    assert classify_state(0.25) == "CRITICAL"  # < 0.30 -> BLOCK
    assert classify_state(0.00) == "CRITICAL"
