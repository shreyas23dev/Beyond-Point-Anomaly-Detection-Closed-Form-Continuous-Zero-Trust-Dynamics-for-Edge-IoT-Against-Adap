"""Model evaluation metrics and plots."""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as plt_sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Utility class for generating evaluation metrics and plots."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use a standard style that exists in most matplotlib installations
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('ggplot')
            
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
        """Compute comprehensive classification metrics."""
        logger.info("Computing evaluation metrics...")
        
        # Calculate PR AUC
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recalls, precisions)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(pr_auc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        
    def compute_per_attack_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        attack_types: pd.Series
    ) -> pd.DataFrame:
        """Compute detection metrics broken down by attack type."""
        logger.info("Computing per-attack metrics...")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        attack_types = np.asarray(attack_types)

        results = []

        for attack in np.unique(attack_types):
            mask = attack_types == attack

            attack_true = y_true[mask]
            attack_pred = y_pred[mask]

            count = len(attack_true)
            if count == 0:
                continue

            # Determine whether this traffic type is attack or normal
            true_label = int(np.bincount(attack_true.astype(int)).argmax())

            if true_label == 1:
                detected = int(np.sum(attack_pred == 1))
                results.append({
                    "Traffic Type": attack,
                    "Count": count,
                    "Detection Rate (Recall)": detected / count,
                    "True Negative Rate": np.nan,
                    "Detected": detected
                })
            else:
                detected = int(np.sum(attack_pred == 0))
                results.append({
                    "Traffic Type": attack,
                    "Count": count,
                    "Detection Rate (Recall)": np.nan,
                    "True Negative Rate": detected / count,
                    "Detected": detected
                })

        return (
            pd.DataFrame(results)
            .sort_values("Count", ascending=False)
            .reset_index(drop=True)
        )
        
    def plot_roc_curve(self, y_true: np.ndarray, scores: np.ndarray, save_path: Path | None = None) -> Path:
        """Plot Receiver Operating Characteristic curve."""
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        save_path = save_path or self.output_dir / "roc_curve.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_pr_curve(self, y_true: np.ndarray, scores: np.ndarray, save_path: Path | None = None) -> Path:
        """Plot Precision-Recall curve."""
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recalls, precisions)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        save_path = save_path or self.output_dir / "pr_curve.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | None = None, normalize: bool = False) -> Path:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
            filename = "confusion_matrix_normalized.png"
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            filename = "confusion_matrix.png"
            
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        plt_sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax, 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        save_path = save_path or self.output_dir / filename
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_score_distribution(self, scores_normal: np.ndarray, scores_attack: np.ndarray, threshold: float | None = None, save_path: Path | None = None) -> Path:
        """Plot histogram of anomaly scores for normal vs attack traffic."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        plt_sns.histplot(scores_normal, bins=50, color='green', alpha=0.5, label='Normal Traffic', ax=ax, stat='density')
        plt_sns.histplot(scores_attack, bins=50, color='red', alpha=0.5, label='Attack Traffic', ax=ax, stat='density')
        
        if threshold is not None:
            ax.axvline(threshold, color='black', linestyle='--', lw=2, label=f'Threshold ({threshold:.3f})')
            
        ax.set_xlabel('Normalized Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Anomaly Score Distribution by Class')
        ax.legend()
        
        save_path = save_path or self.output_dir / "score_distribution.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_per_attack_detection(self, per_attack_df: pd.DataFrame, save_path: Path | None = None) -> Path:
        """Plot detection rates per attack type as a horizontal bar chart."""
        # Filter for rows where metric is "Detection Rate (Recall)"
        if "Detection Rate (Recall)" not in per_attack_df.columns:
            logger.warning("Could not find 'Detection Rate (Recall)' column in dataframe. Skipping plot.")
            return None
            
        attack_df = per_attack_df[per_attack_df["Detection Rate (Recall)"].notna()].copy()
        attack_df = attack_df.sort_values("Detection Rate (Recall)", ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        bars = ax.barh(attack_df["Traffic Type"], attack_df["Detection Rate (Recall)"], color='salmon')
        
        # Add values at the end of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2%}', 
                    ha='left', va='center', fontweight='bold')
            
        ax.set_xlim([0, 1.1])
        ax.set_xlabel('Detection Rate (Recall)')
        ax.set_title('Detection Rate by Attack Type')
        
        save_path = save_path or self.output_dir / "per_attack_detection.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def plot_threshold_comparison(self, comparison_df: pd.DataFrame, save_path: Path | None = None) -> Path:
        """Plot F1, Precision, and Recall for different threshold methods."""
        # Melt DataFrame for seaborn grouped bar chart
        df_melt = pd.melt(comparison_df, id_vars=['method', 'threshold'], 
                          value_vars=['f1', 'precision', 'recall'],
                          var_name='Metric', value_name='Score')
                          
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        plt_sns.barplot(data=df_melt, x='method', y='Score', hue='Metric', ax=ax, palette='muted')
        
        # Add threshold labels above method names
        for i, row in comparison_df.iterrows():
            ax.text(i, -0.05, f"t={row['threshold']:.3f}", ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
            
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score')
        ax.set_xlabel('Calibration Method')
        ax.set_title('Threshold Calibration Method Comparison')
        ax.legend(loc='lower right')
        
        save_path = save_path or self.output_dir / "threshold_comparison.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
        
    def generate_all_plots(
        self, y_true, y_pred, scores, attack_types, 
        threshold, comparison_df, scores_normal, scores_attack
    ) -> list[Path]:
        """Run all plot functions and return list of saved file paths."""
        paths = []
        paths.append(self.plot_roc_curve(y_true, scores))
        paths.append(self.plot_pr_curve(y_true, scores))
        paths.append(self.plot_confusion_matrix(y_true, y_pred))
        paths.append(self.plot_confusion_matrix(y_true, y_pred, normalize=True))
        paths.append(self.plot_score_distribution(scores_normal, scores_attack, threshold))
        
        if attack_types is not None:
            per_attack_df = self.compute_per_attack_metrics(y_true, y_pred, attack_types)
            p = self.plot_per_attack_detection(per_attack_df)
            if p: paths.append(p)
            
        if comparison_df is not None:
            paths.append(self.plot_threshold_comparison(comparison_df))
            
        return paths
