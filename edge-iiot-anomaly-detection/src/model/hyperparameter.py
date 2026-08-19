"""Optuna-based hyperparameter tuning for Isolation Forest."""
import logging

import optuna
import pandas as pd

from src.model.isolation_forest import AnomalyDetector
from src.model.threshold import ThresholdCalibrator

logger = logging.getLogger(__name__)

def tune_hyperparameters(X_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 50, random_state: int = 42) -> dict:
    """Tune Isolation Forest hyperparameters using Optuna.
    
    Args:
        X_train: Training data (only normal traffic).
        X_val: Validation data (mixed traffic).
        y_val: Validation labels.
        n_trials: Number of Optuna trials.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary with best_params, best_f1, all_trials_df, and the study object.
    """
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_samples": trial.suggest_categorical("max_samples", ["auto", 0.5, 0.7, 1.0]),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0, step=0.1),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": random_state,
            "n_jobs": -1
        }
        
        # 1. Train model
        model = AnomalyDetector(**params)
        model.fit(X_train)
        
        # 2. Get validation scores
        val_scores = model.score_samples(X_val)
        
        # 3. Find optimal threshold using F1-optimized
        _, best_f1 = ThresholdCalibrator.f1_optimized_threshold(val_scores, y_val.to_numpy())
        
        return best_f1
        
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner()
    
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Tuning complete. Best F1: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Format trials dataframe
    trials_df = study.trials_dataframe()
    
    # Merge best params with defaults
    best_params = {
        "random_state": random_state,
        "n_jobs": -1
    }
    best_params.update(study.best_params)
    
    return {
        "best_params": best_params,
        "best_f1": study.best_value,
        "all_trials_df": trials_df,
        "study": study
    }
