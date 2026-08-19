"""Core Isolation Forest wrapper."""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config.settings import RANDOM_SEED, config

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Wrapper for scikit-learn IsolationForest with custom normalization and thresholding."""
    
    def __init__(
        self, 
        n_estimators: int = config.model.n_estimators, 
        max_samples: str | float = config.model.max_samples, 
        max_features: float = config.model.max_features, 
        bootstrap: bool = config.model.bootstrap, 
        random_state: int = config.model.random_state, 
        n_jobs: int = config.model.n_jobs
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            # We don't use 'auto' or predefined contamination for decision_function.
            # We rely purely on score_samples for custom thresholding.
        )
        
        self.score_min_: float | None = None
        self.score_max_: float | None = None
        self.threshold_: float | None = None
        self.is_fitted_ = False
        
    def fit(self, X_train: np.ndarray | pd.DataFrame) -> 'AnomalyDetector':
        """Fit Isolation Forest on normal traffic only."""
        logger.info(f"Fitting Isolation Forest on {len(X_train)} samples...")
        self.model.fit(X_train)
        
        # Compute min and max scores on training data for normalization later
        # IF returns negative scores where more negative = more anomalous.
        # We multiply by -1 so higher = more anomalous.
        raw_scores = -1.0 * self.model.score_samples(X_train)
        
        self.score_min_ = float(np.min(raw_scores))
        self.score_max_ = float(np.max(raw_scores))
        
        if self.score_max_ == self.score_min_:
            logger.warning("All training samples received the exact same score. Normalization will fail.")
            # Add a tiny epsilon to avoid div by zero
            self.score_max_ += 1e-10
            
        logger.info(f"Training score range: [{self.score_min_:.4f}, {self.score_max_:.4f}]")
        self.is_fitted_ = True
        return self
        
    def score_samples(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return normalized anomaly scores in [0, 1] range. 1 = most anomalous."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling score_samples.")
            
        raw_scores = -1.0 * self.model.score_samples(X)
        
        # Min-max normalization using training bounds
        normalized = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_)
        
        # Clip to [0, 1] in case inference scores fall outside training bounds
        return np.clip(normalized, 0.0, 1.0)
        
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return binary predictions (1 = anomaly, 0 = normal) based on calibrated threshold."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict.")
        if self.threshold_ is None:
            raise RuntimeError("Threshold must be set before calling predict.")
            
        scores = self.score_samples(X)
        return (scores >= self.threshold_).astype(int)
        
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> list[dict]:
        """Return detailed prediction dicts including confidence."""
        if not self.is_fitted_ or self.threshold_ is None:
            raise RuntimeError("Model must be fitted and threshold set before calling predict_proba.")
            
        scores = self.score_samples(X)
        preds = self.predict(X)
        
        max_possible_distance = max(self.threshold_, 1.0 - self.threshold_)
        
        results = []
        for score, pred in zip(scores, preds):
            distance = abs(score - self.threshold_)
            confidence = min(distance / max_possible_distance, 1.0)
            
            results.append({
                "raw_anomaly_score": float(score),
                "is_anomaly": bool(pred),
                "confidence": float(confidence),
                "anomaly_threshold": float(self.threshold_)
            })
            
        return results
        
    def set_threshold(self, threshold: float):
        """Set the anomaly threshold manually."""
        if not (0.0 <= threshold <= 1.0):
            logger.warning(f"Setting threshold {threshold} outside [0, 1] bounds.")
        self.threshold_ = threshold
        logger.info(f"Anomaly threshold set to {self.threshold_:.4f}")
        
    def get_params_dict(self) -> dict:
        """Get model parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state
        }
        
    def save(self, filepath: Path):
        """Serialize model to disk."""
        logger.info(f"Saving model to {filepath}...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: Path) -> 'AnomalyDetector':
        """Load model from disk."""
        logger.info(f"Loading model from {filepath}...")
        return joblib.load(filepath)
