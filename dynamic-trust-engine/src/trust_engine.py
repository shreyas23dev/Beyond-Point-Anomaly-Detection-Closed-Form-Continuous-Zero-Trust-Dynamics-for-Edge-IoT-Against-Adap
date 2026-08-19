"""Dynamic Trust Engine Core Orchestrator."""
import logging
from datetime import datetime

from src.config_loader import TrustConfig
from src.trust_math.alpha import compute_alpha
from src.trust_math.constraint_validator import (
    validate_post_update,
    validate_pre_update,
    clamp
)
from src.trust_math.decay import compute_decay
from src.trust_math.ema import compute_ema
from src.trust_math.recovery import compute_recovery
from src.trust_math.state_machine import classify_state, log_state_transition
from src.trust_math.threshold import compute_trust_threshold
from src.trust_math.trends import compute_trend
from src.trust_models import DeviceTrustState, TrustExplanation, TrustRequest, TrustResponse
from src.trust_storage import TrustStorage

logger = logging.getLogger(__name__)

class DynamicTrustEngine:
    """Orchestrates the dynamic trust computation pipeline."""
    
    def __init__(self, config_path: str = "config/trust_config.yaml", storage_path: str = "data/trust_history.json"):
        self.config = TrustConfig(config_path)
        self.storage = TrustStorage(storage_path)
        
    def process_request(self, req: TrustRequest) -> tuple[TrustResponse, TrustExplanation]:
        """
        Execute the exact DR-001 Processing Pipeline:
        1. Validate Input (handled by Pydantic TrustRequest)
        2. Load Configuration
        3. Load Trust State
        4. Constraint Validation
        5. Threshold Calculation
        6. Decay OR Recovery
        7. Clamp Trust
        8. Update EMA
        9. Update History
        10. Determine Trend
        11. Determine State
        12. Persist
        13. Return
        """
        logger.info(f"Processing trust update for device {req.device_id} (Type: {req.device_type})")
        
        # 2. Load Configuration
        policy = self.config.get_policy(req.device_type)
        boundaries = self.config.states
        
        # 3. Load Trust State
        state = self.storage.load_device(req.device_id)
        if not state:
            logger.info(f"New device {req.device_id}. Initializing state.")
            initial_trust = float(policy.get("initial_trust", 1.0))
            state = DeviceTrustState(
                device_id=req.device_id,
                device_type=req.device_type,
                trust_score=initial_trust,
                trust_state=classify_state(initial_trust, boundaries),
                ema=initial_trust,
                history=[],
                last_update=req.timestamp,
                last_anomaly_score=req.raw_anomaly_score
            )
            
        trust_before = state.trust_score
            
        # 4. Constraint Validation (Pre-update)
        validate_pre_update(req.raw_anomaly_score, req.anomaly_threshold, trust_before)
        
        # 5. Threshold Calculation
        trust_threshold = compute_trust_threshold(req.anomaly_threshold)
        
        # 6. Decay OR Recovery
        if req.raw_anomaly_score >= req.anomaly_threshold:
            # Attack -> Decay
            new_trust = compute_decay(trust_before, req.raw_anomaly_score)
            formula_used = "Decay"
            reason = f"Anomaly score ({req.raw_anomaly_score:.4f}) exceeded threshold ({req.anomaly_threshold:.4f})"
            alpha_used = 0.0 # not used in decay
        else:
            # Normal -> Recovery
            alpha_used = compute_alpha(
                initial_trust=float(policy.get("initial_trust", 1.0)),
                target_trust=float(policy.get("recovery_target", 0.90)),
                recovery_steps=int(policy.get("recovery_steps", 20))
            )
            new_trust = compute_recovery(trust_before, alpha_used)
            formula_used = "Recovery"
            reason = f"Normal behaviour. Recovering towards {policy.get('recovery_target')}."
            
        # 7. Clamp Trust
        new_trust = clamp(new_trust)
        
        # Additional Constraint Validation (Post-update)
        validate_post_update(new_trust, alpha_used)
        
        # 8. Update EMA (Slow-Burn logic)
        # Note: EMA captures the slow degradation. The spec states "apply an additional trust reduction 
        # even when individual anomaly scores remain below the anomaly threshold if EMA indicates degradation".
        # We'll update the EMA first.
        beta = float(policy.get("ema_beta", 0.8))
        new_ema = compute_ema(new_trust, state.ema, beta)
        
        # Check slow-burn condition: If EMA drops significantly below the current trust or threshold,
        # it indicates a sustained period of lower-than-expected trust.
        # Simplest slow-burn rule: If EMA < trust_threshold, force trust down to EMA.
        if new_ema < trust_threshold and new_trust >= trust_threshold:
            logger.warning(f"Slow-burn detected for {req.device_id}. EMA ({new_ema:.4f}) < Threshold ({trust_threshold:.4f}). Penalizing trust.")
            new_trust = min(new_trust, new_ema)
            reason = "Slow-burn attack detected via EMA degradation."
            
        # 9. Update History
        window_size = int(policy.get("history_window", 50))
        state.history.append(new_trust)
        if len(state.history) > window_size:
            state.history.pop(0) # FIFO
            
        # 10. Determine Trend
        trend = compute_trend(new_trust, trust_before)
        
        # 11. Determine State
        new_state = classify_state(new_trust, boundaries)
        log_state_transition(req.device_id, state.trust_state, new_state, new_trust, req.raw_anomaly_score)
        
        # Update device state object
        state.trust_score = new_trust
        state.ema = new_ema
        state.trust_state = new_state
        state.last_update = req.timestamp
        state.last_anomaly_score = req.raw_anomaly_score
        
        # 12. Persist
        self.storage.save_device(state)
        
        # 13. Return Data Contracts
        response = TrustResponse(
            device_id=req.device_id,
            timestamp=req.timestamp,
            trust_score=new_trust,
            trust_threshold=trust_threshold,
            trust_state=new_state,
            trend=trend,
            reason=reason,
            ema=new_ema
        )
        
        explanation = TrustExplanation(
            device_id=req.device_id,
            timestamp=req.timestamp,
            formula_used=formula_used,
            trust_before=trust_before,
            trust_after=new_trust,
            anomaly_score=req.raw_anomaly_score,
            anomaly_threshold=req.anomaly_threshold,
            trust_threshold=trust_threshold,
            trend=trend,
            reason=reason,
            top_features=req.top_features
        )
        
        return response, explanation
