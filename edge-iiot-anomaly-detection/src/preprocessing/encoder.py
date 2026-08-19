"""Categorical feature encoding transformer."""
import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Transformer for encoding categorical features."""
    
    def __init__(self, handle_unknown: str = 'use_encoded_value', unknown_value: int = -1):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categorical_columns_: list[str] = []
        self.encoders_: dict[str, OrdinalEncoder] = {}
        self.encoding_maps_: dict[str, dict[str, Any]] = {}
        self.feature_names_out_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEncoder':
        """Identify categorical columns and fit encoders."""
        logger.info("Fitting FeatureEncoder...")
        self.feature_names_out_ = list(X.columns)
        
        # Identify categorical columns
        self.categorical_columns_ = [
            col for col in X.columns 
            if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col])
        ]
        
        logger.info(f"Found {len(self.categorical_columns_)} categorical columns to encode: {self.categorical_columns_}")
        
        # Fit an OrdinalEncoder for each categorical column
        for col in self.categorical_columns_:
            # We fit on 2D array as required by sklearn
            encoder = OrdinalEncoder(
                handle_unknown=self.handle_unknown, 
                unknown_value=self.unknown_value,
                dtype=int
            )
            encoder.fit(X[[col]])
            self.encoders_[col] = encoder
            
            # Store mapping for debugging/traceability
            categories = encoder.categories_[0]
            self.encoding_maps_[col] = {str(cat): int(idx) for idx, cat in enumerate(categories)}
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns."""
        if not self.categorical_columns_:
            return X.copy()
            
        logger.info("Transforming data with FeatureEncoder...")
        X_out = X.copy()
        
        for col in self.categorical_columns_:
            if col in X_out.columns:
                # Ensure input is string to match training
                X_out[col] = X_out[col].astype(str)
                X_out[col] = self.encoders_[col].transform(X_out[[col]])
                
        return X_out
        
    def get_encoding_maps(self) -> dict[str, dict[str, Any]]:
        """Return the encoding mappings."""
        return self.encoding_maps_
        
    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
