"""Replay tool for Zero Trust Decision Engine."""
import logging
from pathlib import Path
from typing import List

from src.database import DatabaseManager
from src.zta_models import ZTAInput, ZTAOutput
from src.policy_loader import PolicyLoader
from src.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

class DecisionReplay:
    """Replays historical decisions through a new policy engine version."""
    
    def __init__(self, db_manager: DatabaseManager, policy_engine: PolicyEngine):
        self.db = db_manager
        self.engine = policy_engine
        
    def fetch_history(self, limit: int = 100) -> List[dict]:
        query = "SELECT * FROM policy_log ORDER BY timestamp ASC LIMIT ?"
        return self.db.fetch_all(query, (limit,))
        
    def replay(self, history: List[dict]) -> dict:
        """
        Replay a history of inputs against the loaded engine.
        Returns stats on matches vs mismatches.
        """
        results = {"total": 0, "match": 0, "mismatch": 0, "details": []}
        
        for row in history:
            req = ZTAInput(
                device_id=row['device_id'],
                trust_score=row['trust_score'],
                trust_threshold=row['trust_threshold'],
                trust_state=row['trust_state'],
                trend="STABLE", # default proxy
                reason="Replay input",
                timestamp=row['timestamp'],
                schema_version=row['schema_version']
            )
            
            try:
                decision, reason, policy_id, rule = self.engine.evaluate(req)
                is_match = (decision == row['decision'])
                
                if is_match:
                    results["match"] += 1
                else:
                    results["mismatch"] += 1
                    
                results["details"].append({
                    "audit_id": row['audit_id'],
                    "old_decision": row['decision'],
                    "new_decision": decision,
                    "match": is_match
                })
                
            except Exception as e:
                logger.error(f"Replay failed for row {row['audit_id']}: {e}")
                
            results["total"] += 1
            
        return results
