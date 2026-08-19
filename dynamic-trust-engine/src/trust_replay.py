"""Trust replay engine for debugging and regression testing."""
import json
import logging
from pathlib import Path

from src.trust_engine import DynamicTrustEngine
from src.trust_models import TrustRequest

logger = logging.getLogger(__name__)

class TrustReplayEngine:
    """Replays historical trust updates from a sequence of inputs."""
    
    def __init__(self, engine: DynamicTrustEngine):
        self.engine = engine
        
    def replay(self, requests: list[TrustRequest]):
        """Replay a sequence of requests and return the history of states."""
        logger.info(f"Replaying {len(requests)} trust requests...")
        responses = []
        explanations = []
        
        for req in requests:
            resp, expl = self.engine.process_request(req)
            responses.append(resp)
            explanations.append(expl)
            
        return responses, explanations
        
    def replay_from_json(self, filepath: str | Path):
        """Replay requests stored in a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        requests = [TrustRequest(**item) for item in data]
        return self.replay(requests)
