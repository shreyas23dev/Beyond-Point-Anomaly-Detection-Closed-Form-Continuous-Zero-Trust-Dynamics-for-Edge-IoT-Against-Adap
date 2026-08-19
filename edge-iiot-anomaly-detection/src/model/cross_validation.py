"""Cross-validation for Isolation Forest."""
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.model.isolation_forest import AnomalyDetector
from src.model.threshold import ThresholdCalibrator

logger = logging.getLogger(__name__)

def cross_validate_model(X_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series, model_params: dict, n_folds: int = 5, random_state: int = 42) -> dict:
    """Perform 5-fold cross validation on normal training data.
    
    Procedure:
    In each fold, train Isolation Forest on the fold's normal-only training split.
    Evaluate the trained model on the held-out validation set (which contains both normal and attack traffic).
    Threshold is calibrated per-fold using the F1-optimized method.
    
    Args:
        X_train: Training data (only normal traffic).
        X_val: Validation data (mixed traffic).
        y_val: Validation labels.
        model_params: Dictionary of hyperparameters for AnomalyDetector.
        n_folds: Number of folds.
        random_state: Random seed.
        
    Returns:
        Dictionary containing CV results and metrics.
    """
    logger.info(f"Starting {n_folds}-fold cross validation...")
    
    # We use non-stratified KFold because X_train contains ONLY normal traffic
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    per_fold_metrics = []
    
    # To check stability, we'll store the mean anomaly score of the holdout normal set for each fold
    stability_scores = []
    
    X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    
    for fold, (train_idx, holdout_idx) in enumerate(kf.split(X_train_np)):
        logger.info(f"--- Fold {fold + 1}/{n_folds} ---")
        
        fold_train_X = X_train_np[train_idx]
        fold_holdout_X = X_train_np[holdout_idx]
        
        # 1. Train model on fold's training split
        model = AnomalyDetector(**model_params)
        model.fit(fold_train_X)
        
        # 2. Check stability on fold's holdout split (all normal)
        holdout_scores = model.score_samples(fold_holdout_X)
        stability_scores.append(float(np.mean(holdout_scores)))
        
        # 3. Evaluate on validation set
        val_scores = model.score_samples(X_val)
        
        # 4. Calibrate threshold using F1-optimized method
        best_threshold, best_f1 = ThresholdCalibrator.f1_optimized_threshold(val_scores, y_val.to_numpy())
        model.set_threshold(best_threshold)
        
        # 5. Get predictions and compute other metrics
        from sklearn.metrics import precision_score, recall_score
        preds = model.predict(X_val)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        
        metrics = {
            "fold": fold + 1,
            "f1": float(best_f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": float(best_threshold)
        }
        per_fold_metrics.append(metrics)
        logger.info(f"Fold {fold + 1} metrics: F1={best_f1:.4f}, P={precision:.4f}, R={recall:.4f}, Thresh={best_threshold:.4f}")
        
    # Aggregate metrics
    f1s = [m['f1'] for m in per_fold_metrics]
    precisions = [m['precision'] for m in per_fold_metrics]
    recalls = [m['recall'] for m in per_fold_metrics]
    
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))
    
    # Stability: variance of mean normal scores across folds (lower is better)
    score_stability = float(np.var(stability_scores))
    
    summary_str = f"CV Results: F1 = {mean_f1:.4f} ± {std_f1:.4f} | Stability (var) = {score_stability:.6f}"
    logger.info(summary_str)
    
    return {
        "per_fold_metrics": per_fold_metrics,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "score_stability": score_stability,
        "summary": summary_str
    }
