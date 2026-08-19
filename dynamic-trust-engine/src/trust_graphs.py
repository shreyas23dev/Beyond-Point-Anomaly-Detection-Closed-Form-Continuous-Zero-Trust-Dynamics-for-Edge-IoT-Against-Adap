"""Graph generation for trust metrics."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as plt_sns
from src.trust_storage import TrustStorage

logger = logging.getLogger(__name__)

class TrustGraphGenerator:
    """Generates visual artifacts for trust history."""
    
    def __init__(self, storage_path: str = "data/trust_history.json", output_dir: str = "artifacts"):
        self.storage = TrustStorage(storage_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('ggplot')
            
    def generate_trust_history_graph(self, device_id: str):
        """Generate a line plot of trust history."""
        state = self.storage.load_device(device_id)
        if not state or not state.history:
            logger.warning(f"No history found for device {device_id}")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(state.history, marker='o', linestyle='-', color='blue', label='Trust Score')
        
        # Add EMA line if we had historical EMA, but we only store current EMA.
        # Alternatively we can plot the current EMA as a horizontal reference line for now.
        ax.axhline(state.ema, color='red', linestyle='--', label=f'Current EMA ({state.ema:.2f})')
        
        ax.set_ylim([-0.05, 1.05])
        ax.set_title(f"Trust History for {device_id}")
        ax.set_xlabel("Time (Updates)")
        ax.set_ylabel("Trust Score")
        ax.legend()
        
        save_path = self.output_dir / f"trust_history_{device_id}.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved trust history graph to {save_path}")
