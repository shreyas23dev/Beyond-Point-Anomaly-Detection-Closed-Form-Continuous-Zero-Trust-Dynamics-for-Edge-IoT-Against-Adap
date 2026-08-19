"""Data cleaning transformer."""
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.settings import config

logger = logging.getLogger(__name__)

class DataCleaner(BaseEstimator, TransformerMixin):
    """Transformer for cleaning the Edge-IIoTset dataset."""
    
    def __init__(self, columns_to_drop: list[str] | None = None, target_columns: list[str] | None = None):
        """Initialize DataCleaner.
        
        Args:
            columns_to_drop: List of columns to drop (defaults to config).
            target_columns: List of target/label columns to drop before engineering.
        """
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else config.dataset.columns_to_drop
        self.target_columns = target_columns if target_columns is not None else [config.dataset.attack_type_column, config.dataset.target_column]
        self.actual_columns_to_drop_: list[str] = []
        self.imputation_values_: dict[str, Any] = {}
        self.feature_names_out_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y=None) -> 'DataCleaner':
        """Learn imputation values from training data."""
        logger.info("Fitting DataCleaner...")
        
        # Determine which columns to drop actually exist in X
        all_to_drop = self.columns_to_drop + self.target_columns
        self.actual_columns_to_drop_ = [col for col in all_to_drop if col in X.columns]
        
        if len(self.actual_columns_to_drop_) < len(all_to_drop):
            missing = set(all_to_drop) - set(self.actual_columns_to_drop_)
            logger.info(f"Some columns to drop were not found in data: {missing}")
            
        # Determine remaining columns
        remaining_cols = [col for col in X.columns if col not in self.actual_columns_to_drop_]
        
        # Learn imputation values
        for col in remaining_cols:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Use median for numeric
                val = X[col].replace([np.inf, -np.inf], np.nan).median()
                # Fallback to 0 if all are NaN
                self.imputation_values_[col] = 0 if pd.isna(val) else val
            else:
                # Use mode for categorical
                mode_series = X[col].mode()
                val = mode_series.iloc[0] if not mode_series.empty else "unknown"
                self.imputation_values_[col] = val
                
        self.feature_names_out_ = remaining_cols
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations."""
        logger.info(f"Transforming data with DataCleaner (input shape: {X.shape})...")
        X_out = X.copy()
        
        # 1. Drop columns
        to_drop = [c for c in self.actual_columns_to_drop_ if c in X_out.columns]
        if to_drop:
            X_out = X_out.drop(columns=to_drop)
            
        # 2. Replace inf with NaN
        X_out = X_out.replace([np.inf, -np.inf], np.nan)
        
        # 3. Impute missing values
        for col, val in self.imputation_values_.items():
            if col in X_out.columns:
                if X_out[col].isna().any():
                    X_out[col] = X_out[col].fillna(val)
                    
        # 4. Type casting (ensure object columns are string for encoder)
        for col in X_out.columns:
            if X_out[col].dtype == 'object':
                X_out[col] = X_out[col].astype(str)
                
        # 5. Ensure column order matches fit
        # During inference, we might only have a subset of columns, or they might be out of order.
        # But we expect the pipeline to receive full data (minus targets).
        # We just reorder to match if they all exist.
        if all(c in X_out.columns for c in self.feature_names_out_):
            X_out = X_out[self.feature_names_out_]
            
        logger.info(f"DataCleaner transform complete (output shape: {X_out.shape}).")
        return X_out
        
    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
