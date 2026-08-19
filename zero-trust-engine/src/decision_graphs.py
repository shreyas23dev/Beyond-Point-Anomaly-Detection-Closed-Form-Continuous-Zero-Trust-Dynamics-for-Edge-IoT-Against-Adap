"""Graph generation for Zero Trust Engine."""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

class DecisionGraphGenerator:
    """Generates visualizations for reports and presentations."""
    
    def __init__(self, db_manager: DatabaseManager, output_dir: Path = Path("artifacts")):
        self.db = db_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all(self):
        self.generate_allow_verify_block()
        self.generate_policy_usage()
        
    def generate_allow_verify_block(self):
        """Generates allow_verify_block.png"""
        stats = self.db.get_decision_stats()
        labels = ['ALLOW', 'VERIFY', 'BLOCK']
        counts = [stats.get('allow', 0), stats.get('verify', 0), stats.get('block', 0)]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color=colors)
        plt.title('Zero Trust Decisions Distribution')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        filepath = self.output_dir / "allow_verify_block.png"
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Saved graph to {filepath}")

    def generate_policy_usage(self):
        """Generates policy_usage.png"""
        query = "SELECT policy_id, COUNT(*) as count FROM policy_log GROUP BY policy_id"
        rows = self.db.fetch_all(query)
        if not rows:
            return
            
        labels = [r['policy_id'] for r in rows]
        counts = [r['count'] for r in rows]
        
        plt.figure(figsize=(10, 6))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Policy Usage Distribution')
        
        filepath = self.output_dir / "policy_usage.png"
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Saved graph to {filepath}")
