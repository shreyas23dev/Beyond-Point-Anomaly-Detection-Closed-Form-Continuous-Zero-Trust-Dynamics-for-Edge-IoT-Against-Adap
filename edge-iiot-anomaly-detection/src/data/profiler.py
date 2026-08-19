"""Dataset profiling utility."""
import json
import logging
from pathlib import Path

import pandas as pd

try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False

from config.settings import config

logger = logging.getLogger(__name__)

class DatasetProfiler:
    """Utility class for dataset profiling and EDA."""
    
    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_basic_stats(self) -> dict:
        """Generate basic dataset statistics."""
        logger.info("Generating basic dataset statistics...")
        stats = {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_pct": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            "unique_counts": self.df.nunique().to_dict(),
        }
        return stats
        
    def analyze_class_distribution(self) -> dict:
        """Analyze attack vs normal distribution."""
        logger.info("Analyzing class distribution...")
        dist = {}
        if config.dataset.attack_type_column in self.df.columns:
            dist["attack_type_counts"] = self.df[config.dataset.attack_type_column].value_counts().to_dict()
            
        if config.dataset.target_column in self.df.columns:
            target_counts = self.df[config.dataset.target_column].value_counts()
            dist["target_counts"] = target_counts.to_dict()
            if len(target_counts) == 2:
                normal = target_counts.get(config.dataset.normal_label, 0)
                attack = target_counts.get(config.dataset.attack_label, 0)
                total = normal + attack
                if total > 0:
                    dist["normal_ratio"] = float(normal / total)
                    dist["attack_ratio"] = float(attack / total)
                    
        return dist
        
    def analyze_feature_distributions(self) -> dict:
        """Analyze numeric feature distributions (skewness, kurtosis)."""
        logger.info("Analyzing numeric feature distributions...")
        numeric_df = self.df.select_dtypes(include=["number"])
        
        # Exclude target if it's numeric
        if config.dataset.target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[config.dataset.target_column])
            
        distributions = {}
        if not numeric_df.empty:
            distributions["skewness"] = numeric_df.skew().fillna(0).to_dict()
            distributions["kurtosis"] = numeric_df.kurtosis().fillna(0).to_dict()
            
        return distributions
        
    def generate_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """Generate correlation matrix for numeric features."""
        logger.info(f"Generating {method} correlation matrix...")
        numeric_df = self.df.select_dtypes(include=["number"])
        
        # Drop columns with 0 variance before correlation
        variances = numeric_df.var()
        numeric_df = numeric_df.loc[:, variances > 0]
        
        return numeric_df.corr(method=method)
        
    def generate_html_report(self) -> Path | None:
        """Generate ydata-profiling HTML report."""
        if not YDATA_AVAILABLE:
            logger.warning("ydata-profiling not installed. Skipping HTML report generation.")
            return None
            
        logger.info("Generating ydata-profiling HTML report (this may take a while)...")
        report_path = self.output_dir / "dataset_profile_report.html"
        
        try:
            profile = ProfileReport(self.df, title="Edge-IIoTset Dataset Profiling Report", minimal=True)
            profile.to_file(report_path)
            logger.info(f"HTML report saved to {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return None
            
    def generate_summary(self) -> dict:
        """Combine all statistics into a single summary dictionary."""
        summary = {
            "basic_stats": self.generate_basic_stats(),
            "class_distribution": self.analyze_class_distribution(),
            "feature_distributions": self.analyze_feature_distributions(),
        }
        return summary
        
    def save_profile(self, summary: dict) -> Path:
        """Save the profile summary as a JSON file."""
        output_path = self.output_dir / "dataset_profile.json"
        
        # Convert any potential numpy types to native Python types for JSON serialization
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
            
        clean_summary = convert_numpy(summary)
        
        with open(output_path, "w") as f:
            json.dump(clean_summary, f, indent=2)
            
        logger.info(f"Dataset profile saved to {output_path}")
        return output_path
