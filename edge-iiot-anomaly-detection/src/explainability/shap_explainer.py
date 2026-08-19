"""SHAP values computation and plotting for Isolation Forest."""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

class TreeShapExplainer:
    """Wrapper for SHAP TreeExplainer for Isolation Forest models."""
    
    def __init__(self, model, output_dir: Path):
        """Initialize SHAP explainer.
        
        Args:
            model: Fitted Isolation Forest model (the underlying sklearn model).
            output_dir: Directory to save plots.
        """
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing SHAP TreeExplainer...")
        # TreeExplainer works directly on sklearn IsolationForest
        self.explainer = shap.TreeExplainer(self.model)
        
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the dataset.
        
        Note: For IsolationForest, SHAP returns expected anomaly scores (path lengths).
        Since we flip scores for our API, we should flip the SHAP values too, so positive
        SHAP values contribute to HIGHER anomaly scores (more anomalous).
        
        Args:
            X: Input features dataframe.
            
        Returns:
            Numpy array of SHAP values.
        """
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        # SHAP values for IsolationForest explain the raw score (path length).
        # We negate them so that positive SHAP = more anomalous.
        shap_values = -1.0 * self.explainer.shap_values(X)
        return shap_values
        
    def plot_summary(self, shap_values: np.ndarray, X: pd.DataFrame, max_display: int = 20, save_path: Path | None = None) -> Path:
        """Generate SHAP summary plot (beeswarm)."""
        logger.info("Generating SHAP summary plot...")
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)), dpi=150)
        
        shap.summary_plot(
            shap_values, X, 
            plot_type="dot", 
            max_display=max_display, 
            show=False,
            plot_size="auto"
        )
        
        plt.title('SHAP Summary Plot (Top Features)')
        
        save_path = save_path or self.output_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
        
    def plot_feature_importance(self, shap_values: np.ndarray, X: pd.DataFrame, max_display: int = 20, save_path: Path | None = None) -> Path:
        """Generate SHAP feature importance plot (bar chart of mean |SHAP|)."""
        logger.info("Generating SHAP feature importance plot...")
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)), dpi=150)
        
        shap.summary_plot(
            shap_values, X, 
            plot_type="bar", 
            max_display=max_display, 
            show=False,
            plot_size="auto"
        )
        
        plt.title('SHAP Feature Importance (Mean |SHAP Value|)')
        
        save_path = save_path or self.output_dir / "shap_importance.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
        
    def get_global_importance_dict(self, shap_values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
        """Get mean absolute SHAP value for each feature."""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_dict = {
            feat: float(val) for feat, val in zip(feature_names, mean_abs_shap)
        }
        
        # Sort by importance descending
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
    def generate_local_explanation(self, shap_values: np.ndarray, feature_names: list[str], sample_idx: int = 0) -> dict[str, float]:
        """Get local feature contributions for a specific sample."""
        sample_shap = shap_values[sample_idx]
        
        explanation = {
            feat: float(val) for feat, val in zip(feature_names, sample_shap)
        }
        
        # Sort by absolute contribution descending
        return dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
        
    def export_global_importance(self, shap_values: np.ndarray, feature_names: list[str], save_path: Path | None = None) -> Path:
        """Export global feature importance to JSON."""
        importance_dict = self.get_global_importance_dict(shap_values, feature_names)
        
        save_path = save_path or self.output_dir / "shap_feature_importance.json"
        with open(save_path, 'w') as f:
            json.dump(importance_dict, f, indent=2)
            
        logger.info(f"Exported SHAP feature importance to {save_path}")
        return save_path
