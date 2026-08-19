"""Unit tests for trust math modules."""
import pytest
from src.trust_math.threshold import compute_trust_threshold
from src.trust_math.alpha import compute_alpha
from src.trust_math.decay import compute_decay
from src.trust_math.recovery import compute_recovery
from src.trust_math.ema import compute_ema
from src.trust_math.trends import compute_trend
from src.trust_math.state_machine import classify_state
from src.trust_math.constraint_validator import validate_bounds, clamp, ConstraintViolationError

def test_threshold():
    assert compute_trust_threshold(0.2) == pytest.approx(0.8)
    assert compute_trust_threshold(0.0) == pytest.approx(1.0)
    assert compute_trust_threshold(1.0) == pytest.approx(0.0)

def test_alpha():
    # If target is 0.9, T_0 is 0.0 (fallback from 1.0), steps = 10
    # alpha = 1 - ( (1-0.9)/(1-0.0) )^(1/10) = 1 - (0.1)^0.1 = 1 - 0.7943 = 0.2056
    alpha = compute_alpha(1.0, 0.9, 10)
    assert 0.0 < alpha < 1.0
    assert alpha == pytest.approx(0.2056, abs=1e-3)
    
    # Starting from 0.5 to target 0.9 in 5 steps
    # alpha = 1 - (0.1/0.5)^0.2 = 1 - 0.2^0.2 = 1 - 0.7247 = 0.2752
    alpha2 = compute_alpha(0.5, 0.9, 5)
    assert alpha2 == pytest.approx(0.2752, abs=1e-3)

def test_decay():
    # T = 0.8, A = 0.5 -> 0.8 * 0.5 = 0.4
    assert compute_decay(0.8, 0.5) == pytest.approx(0.4)
    # T = 0.8, A = 0.0 -> 0.8
    assert compute_decay(0.8, 0.0) == pytest.approx(0.8)

def test_recovery():
    # T = 0.5, alpha = 0.2 -> 0.5 + 0.2*0.5 = 0.6
    assert compute_recovery(0.5, 0.2) == pytest.approx(0.6)
    
def test_ema():
    # T = 0.8, EMA = 0.9, beta = 0.8 -> 0.8*0.8 + 0.2*0.9 = 0.64 + 0.18 = 0.82
    assert compute_ema(0.8, 0.9, 0.8) == pytest.approx(0.82)
    
def test_trends():
    assert compute_trend(0.8, 0.5) == "RECOVERING"
    assert compute_trend(0.4, 0.5) == "DECLINING"
    assert compute_trend(0.5, 0.5) == "STABLE"

def test_state_machine():
    bounds = {"HIGH": 0.8, "MEDIUM": 0.6, "LOW": 0.4}
    assert classify_state(0.9, bounds) == "HIGH"
    assert classify_state(0.8, bounds) == "HIGH"
    assert classify_state(0.7, bounds) == "MEDIUM"
    assert classify_state(0.5, bounds) == "LOW"
    assert classify_state(0.3, bounds) == "CRITICAL"

def test_constraint_validator():
    validate_bounds(0.5, "test")
    with pytest.raises(ConstraintViolationError):
        validate_bounds(1.5, "test")
    with pytest.raises(ConstraintViolationError):
        validate_bounds(-0.1, "test")
        
    assert clamp(1.5) == 1.0
    assert clamp(-0.5) == 0.0
    assert clamp(0.5) == 0.5
