"""Unified preprocessing pipeline."""
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import FeatureEncoder
from src.preprocessing.scaler import FeatureScaler

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Unified pipeline for cleaning, encoding, and scaling."""
    
    def __init__(self, columns_to_drop=None, target_columns=None, scaling_strategy='robust'):
        self.pipeline = Pipeline([
            ('cleaner', DataCleaner(columns_to_drop=columns_to_drop, target_columns=target_columns)),
            ('encoder', FeatureEncoder()),
            ('scaler', FeatureScaler(strategy=scaling_strategy))
        ])
        self.is_fitted_ = False
        
    def fit(self, X: pd.DataFrame) -> 'PreprocessingPipeline':
        """Fit all components sequentially."""
        logger.info("Fitting PreprocessingPipeline...")
        self.pipeline.fit(X)
        self.is_fitted_ = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        return self.pipeline.transform(X)
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        logger.info("Fit-transforming PreprocessingPipeline...")
        X_out = self.pipeline.fit_transform(X)
        self.is_fitted_ = True
        return X_out
        
    def get_feature_names(self) -> list[str]:
        """Return final feature column names."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted first.")
        # The scaler is the last step, so we get names from it
        return self.pipeline.named_steps['scaler'].get_feature_names_out()
        
    def save(self, filepath: Path):
        """Serialize pipeline to disk."""
        logger.info(f"Saving pipeline to {filepath}...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: Path) -> 'PreprocessingPipeline':
        """Load pipeline from disk."""
        logger.info(f"Loading pipeline from {filepath}...")
        return joblib.load(filepath)
        
    def get_feature_columns_json(self) -> dict:
        """Return metadata about features and transformations."""
        if not self.is_fitted_:
            return {}
            
        cleaner = self.pipeline.named_steps['cleaner']
        encoder = self.pipeline.named_steps['encoder']
        
        return {
            "dropped_columns": cleaner.actual_columns_to_drop_,
            "imputed_columns": list(cleaner.imputation_values_.keys()),
            "categorical_columns": encoder.categorical_columns_,
            "final_feature_columns": self.get_feature_names(),
            "n_features": len(self.get_feature_names())
        }
