"""Dynamic Trust Engine Core Orchestrator (Algorithm 1 Implementation)."""
import logging
from datetime import datetime

from src.config_loader import TrustConfig
from src.trust_math.alpha import compute_alpha
from src.trust_math.decay import compute_decay
from src.trust_math.recovery import compute_recovery
from src.trust_math.ema import compute_anomaly_ema
from src.trust_math.accumulator import compute_slow_burn_accumulator
from src.trust_math.slow_burn_mitigation import compute_slow_burn_mitigation
from src.trust_math.state_machine import classify_state
from src.trust_models import DeviceTrustState, TrustExplanation, TrustRequest, TrustResponse
from src.trust_storage import TrustStorage

logger = logging.getLogger(__name__)

class DynamicTrustEngine:
    """Orchestrates the per-request Zero-Trust computation pipeline (Algorithm 1)."""
    
    def __init__(self, config_path: str = "config/trust_config.yaml", storage_path: str = "data/trust_history.json"):
        self.config = TrustConfig(config_path)
        self.storage = TrustStorage(storage_path)
        
    def process_request(self, req: TrustRequest) -> tuple[TrustResponse, TrustExplanation]:
        """
        Execute Algorithm 1: Per-Request Trust State Update:
        1. Anomaly EMA Update (E_{t+1})
        2. Slow-Burn Evidence Accumulation (SB_{t+1})
        3. Slow-Burn Indicator Evaluation (I_t)
        4. Mutually Exclusive Trust State Transition (Eq. 10)
        5. Qualitative State Assignment (Eq. 11)
        6. Deterministic Audit Record Generation (A_t)
        """
        logger.info(f"Processing trust update for device {req.device_id} (Type: {req.device_type})")
        
        # Load configuration
        policy = self.config.get_policy(req.device_type)
        boundaries = self.config.states
        sb_cfg = self.config.slow_burn
        
        beta = float(sb_cfg.get("ema_beta", 0.40))
        gamma = float(sb_cfg.get("accumulation_threshold_gamma", 0.335))
        lambda_ = float(sb_cfg.get("accumulation_step_lambda", 0.55))
        delta = float(sb_cfg.get("decay_step_delta", 0.05))
        theta = float(sb_cfg.get("trigger_threshold_theta", 1.00))
        penalty_p = float(sb_cfg.get("penalty_weight_p", 0.25))
        
        # Load existing device state or initialize
        state = self.storage.load_device(req.device_id)
        if not state:
            logger.info(f"New device {req.device_id}. Initializing state.")
            initial_trust = float(policy.get("initial_trust", 1.0))
            state = DeviceTrustState(
                device_id=req.device_id,
                device_type=req.device_type,
                trust_score=initial_trust,
                anomaly_ema=0.0,
                slow_burn_score=0.0,
                slow_burn_detected=False,
                trust_state=classify_state(initial_trust, boundaries),
                history=[],
                last_update=req.timestamp,
                last_anomaly_score=req.raw_anomaly_score,
                consecutive_clean_steps=0
            )
            
        trust_before = state.trust_score
        ema_before = getattr(state, "anomaly_ema", 0.0)
        sb_before = getattr(state, "slow_burn_score", 0.0)
        state_before = state.trust_state
        
        # Step 1: Anomaly EMA Energy Update (Eq. 8)
        new_ema = compute_anomaly_ema(
            anomaly_score=req.raw_anomaly_score,
            previous_ema=ema_before,
            beta=beta
        )
        
        # Step 2: Slow-Burn Evidence Accumulation (Eq. 9)
        new_sb, is_slow_burn = compute_slow_burn_accumulator(
            anomaly_ema=new_ema,
            previous_sb=sb_before,
            gamma=gamma,
            lambda_=lambda_,
            delta=delta,
            theta=theta
        )
        
        # Step 3 & 4: Mutually Exclusive Trust State Transition (Eq. 10)
        a_t = req.raw_anomaly_score
        a_thr = req.anomaly_threshold
        
        if a_t >= a_thr:
            # Branch 1: Overt Attack Decay
            new_trust = compute_decay(current_trust=trust_before, anomaly_score=a_t)
            formula_used = "OVERT_DECAY"
            reason = f"Overt anomaly breach (A_t={a_t:.4f} >= A_thr={a_thr:.4f}). Applied multiplicative decay."
            consec_clean = 0
        elif a_t < a_thr and not is_slow_burn:
            # Branch 2: Benign Policy Recovery
            target_trust = float(policy.get("recovery_target", 0.90))
            recovery_steps = int(policy.get("recovery_steps", 10))
            alpha_val = compute_alpha(target_trust=target_trust, recovery_steps=recovery_steps)
            new_trust = compute_recovery(current_trust=trust_before, alpha=alpha_val)
            formula_used = "BENIGN_RECOVERY"
            reason = f"Nominal telemetry (A_t={a_t:.4f} < A_thr={a_thr:.4f}, I_t=0). Applied asymptotic recovery."
            consec_clean = state.consecutive_clean_steps + 1
        else:
            # Branch 3: Slow-Burn Mitigation Penalty
            new_trust = compute_slow_burn_mitigation(
                current_trust=trust_before,
                anomaly_score=a_t,
                anomaly_threshold=a_thr,
                penalty_p=penalty_p
            )
            formula_used = "SLOW_BURN_MITIGATION"
            reason = f"Sustained slow-burn pressure (E_t={new_ema:.4f} > gamma={gamma:.4f}, SB_t={new_sb:.4f} > theta={theta:.2f}). Enforcing progressive penalty."
            consec_clean = 0
            
        # Bounds enforcement
        new_trust = float(max(0.0, min(1.0, new_trust)))
        
        # Step 5: Qualitative State Assignment (Eq. 11)
        new_state = classify_state(new_trust, boundaries)
        
        # Determine trend
        if new_trust > trust_before + 1e-4:
            trend = "RECOVERING"
        elif new_trust < trust_before - 1e-4:
            trend = "DECLINING"
        else:
            trend = "STABLE"
            
        # Update history
        history = state.history[-int(policy.get("history_window", 30)):] + [new_trust]
        
        # Update persisted state
        updated_state = DeviceTrustState(
            device_id=req.device_id,
            device_type=req.device_type,
            trust_score=new_trust,
            anomaly_ema=new_ema,
            slow_burn_score=new_sb,
            slow_burn_detected=is_slow_burn,
            trust_state=new_state,
            history=history,
            last_update=req.timestamp,
            last_anomaly_score=a_t,
            consecutive_clean_steps=consec_clean
        )
        self.storage.save_device(updated_state)
        
        # Step 6: Deterministic Audit Record Generation
        response = TrustResponse(
            device_id=req.device_id,
            timestamp=req.timestamp,
            trust_score=new_trust,
            anomaly_ema=new_ema,
            slow_burn_score=new_sb,
            slow_burn_detected=is_slow_burn,
            trust_threshold=0.60,
            trust_state=new_state,
            trend=trend,
            formula_used=formula_used,
            reason=reason
        )
        
        explanation = TrustExplanation(
            device_id=req.device_id,
            timestamp=req.timestamp,
            formula_used=formula_used,
            trust_before=trust_before,
            trust_after=new_trust,
            anomaly_score=a_t,
            anomaly_ema=new_ema,
            slow_burn_score=new_sb,
            slow_burn_detected=is_slow_burn,
            anomaly_threshold=a_thr,
            trust_state_before=state_before,
            trust_state_after=new_state,
            trend=trend,
            reason=reason,
            top_features=req.top_features
        )
        
        return response, explanation
