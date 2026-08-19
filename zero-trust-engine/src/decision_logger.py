"""Logs decisions to the SQLite database via DatabaseManager."""
import uuid
import logging
from src.database import DatabaseManager
from src.zta_models import ZTAOutput

logger = logging.getLogger(__name__)

class DecisionLogger:
    """Handles creating audit records and saving them."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def log_decision(self, out: ZTAOutput) -> None:
        """Log a ZTAOutput record to the database."""
        query = '''
            INSERT INTO policy_log (
                audit_id, timestamp, device_id, trust_score, trust_threshold, 
                trust_state, policy_id, decision, reason, schema_version, policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        # We need trust_score, trust_threshold, trust_state which are NOT in ZTAOutput
        # Ah, the ZTAOutput only has phase 4 fields. We should pass the input data here too
        # to fulfill the Audit Schema requirements.
        pass
        
    def log_full_audit(self, out: ZTAOutput, req_trust_score: float, req_trust_threshold: float, req_trust_state: str):
        """Log the full audit record."""
        query = '''
            INSERT INTO policy_log (
                audit_id, timestamp, device_id, trust_score, trust_threshold, 
                trust_state, policy_id, decision, reason, schema_version, policy_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            out.audit_id,
            out.timestamp,
            out.device_id,
            req_trust_score,
            req_trust_threshold,
            req_trust_state,
            out.policy_id,
            out.zta_decision,
            out.reason,
            out.schema_version,
            out.policy_version
        )
        self.db.execute_transaction(query, params)
        logger.info(f"AUDIT LOGGED | Audit_ID: {out.audit_id} | Device: {out.device_id} | Decision: {out.zta_decision}")
