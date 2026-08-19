"""Trust state persistence."""
import json
import logging
import threading
from pathlib import Path

from src.trust_models import DeviceTrustState

logger = logging.getLogger(__name__)

class TrustStorage:
    """Thread-safe JSON file storage for device trust states."""
    
    def __init__(self, file_path: str | Path = "data/trust_history.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # Initialize empty file if not exists
        if not self.file_path.exists():
            self._write_raw({})
            
    def _write_raw(self, data: dict):
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def _read_raw(self) -> dict:
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def load_device(self, device_id: str) -> DeviceTrustState | None:
        """Load state for a specific device."""
        with self._lock:
            data = self._read_raw()
            device_data = data.get(device_id)
            if device_data:
                return DeviceTrustState(**device_data)
            return None

    def save_device(self, state: DeviceTrustState):
        """Save state for a specific device."""
        with self._lock:
            data = self._read_raw()
            data[state.device_id] = state.model_dump()
            self._write_raw(data)
