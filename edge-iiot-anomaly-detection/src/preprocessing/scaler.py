"""Feature scaling transformer."""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

class FeatureScaler(BaseEstimator, TransformerMixin):
    """Transformer for scaling numeric features."""
    
    def __init__(self, strategy: str = 'robust'):
        """Initialize FeatureScaler.
        
        Args:
            strategy: 'robust', 'standard', or 'minmax'
        """
        self.strategy = strategy
        self.numeric_columns_: list[str] = []
        self.feature_names_out_: list[str] = []
        
        if strategy == 'robust':
            self.scaler_ = RobustScaler()
        elif strategy == 'standard':
            self.scaler_ = StandardScaler()
        elif strategy == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
            
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureScaler':
        """Fit scaler on numeric columns."""
        logger.info(f"Fitting FeatureScaler (strategy: {self.strategy})...")
        self.feature_names_out_ = list(X.columns)
        
        # Identify numeric columns
        self.numeric_columns_ = [
            col for col in X.columns 
            if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col])
        ]
        
        logger.info(f"Found {len(self.numeric_columns_)} numeric columns to scale.")
        
        if self.numeric_columns_:
            self.scaler_.fit(X[self.numeric_columns_])
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns."""
        if not self.numeric_columns_:
            return X.copy()
            
        logger.info("Transforming data with FeatureScaler...")
        X_out = X.copy()
        
        cols_to_scale = [c for c in self.numeric_columns_ if c in X_out.columns]
        if cols_to_scale:
            scaled_values = self.scaler_.transform(X_out[cols_to_scale])
            X_out[cols_to_scale] = scaled_values
            
        return X_out
        
    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
