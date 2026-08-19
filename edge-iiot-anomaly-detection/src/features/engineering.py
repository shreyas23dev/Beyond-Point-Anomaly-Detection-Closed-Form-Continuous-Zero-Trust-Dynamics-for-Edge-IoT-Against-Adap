"""Feature engineering transformer."""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer for engineering new features."""
    
    def __init__(self, create_ratios: bool = True, create_log_transforms: bool = True, create_interactions: bool = True):
        self.create_ratios = create_ratios
        self.create_log_transforms = create_log_transforms
        self.create_interactions = create_interactions
        self.engineered_feature_names_: list[str] = []
        self.source_columns_: list[str] = []
        self.skewed_columns_: list[str] = []
        self.interaction_pairs_: list[tuple[str, str]] = []
        self.feature_names_out_: list[str] = []
        
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        """Identify which features can be created."""
        logger.info("Fitting FeatureEngineer...")
        self.source_columns_ = list(X.columns)
        self.feature_names_out_ = list(X.columns)
        
        numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        
        # 1. Identify log transforms
        if self.create_log_transforms and numeric_cols:
            # Only consider positive columns for log (though we use log1p(abs(x)))
            # We look for high skewness
            skewness = X[numeric_cols].skew()
            self.skewed_columns_ = skewness[skewness > 2.0].index.tolist()
            logger.info(f"Identified {len(self.skewed_columns_)} skewed columns for log transform.")
            for col in self.skewed_columns_:
                new_col = f"{col}_log"
                self.engineered_feature_names_.append(new_col)
                self.feature_names_out_.append(new_col)
                
        # 2. Identify ratios (based on typical IIoT column names)
        # We just check if the source columns exist.
        if self.create_ratios:
            # We don't need to fit anything specific for ratios, we just know what we want to create
            # and we'll check existence during transform. However, we can add them to feature_names_out_ here.
            potential_ratios = [
                ('tcp.len', 'udp.length', 'tcp_udp_ratio'),
                ('tcp.payload_len', 'frame.len', 'payload_header_ratio'),
                ('frame.len', 'frame.time_delta', 'flow_rate')
            ]
            for num, den, name in potential_ratios:
                if num in self.source_columns_ and den in self.source_columns_:
                    self.engineered_feature_names_.append(name)
                    self.feature_names_out_.append(name)
                    
        # 3. Identify interactions (top correlated pairs)
        if self.create_interactions and len(numeric_cols) > 1:
            # Compute correlation matrix, find top 5 highly correlated (but not identical) pairs
            corr = X[numeric_cols].corr().abs()
            # Mask upper triangle to avoid duplicates and self-correlation
            mask = np.triu(np.ones_like(corr, dtype=bool))
            corr_masked = corr.mask(mask)
            
            # Unstack and sort
            pairs = corr_masked.unstack().dropna().sort_values(ascending=False)
            
            # Take top 5 pairs where correlation is < 0.99 (to avoid redundant features)
            top_pairs = pairs[pairs < 0.99].head(5).index.tolist()
            self.interaction_pairs_ = top_pairs
            logger.info(f"Identified {len(self.interaction_pairs_)} interaction pairs.")
            
            for col1, col2 in self.interaction_pairs_:
                new_col = f"{col1}_x_{col2}"
                self.engineered_feature_names_.append(new_col)
                self.feature_names_out_.append(new_col)
                
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create the engineered features."""
        logger.info("Transforming data with FeatureEngineer...")
        X_out = X.copy()
        
        # 1. Log transforms
        if self.create_log_transforms:
            for col in self.skewed_columns_:
                if col in X_out.columns:
                    new_col = f"{col}_log"
                    X_out[new_col] = np.log1p(np.abs(X_out[col]))
                    
        # 2. Ratios
        if self.create_ratios:
            epsilon = 1e-10
            
            if 'tcp.len' in X_out.columns and 'udp.length' in X_out.columns:
                X_out['tcp_udp_ratio'] = X_out['tcp.len'] / (X_out['udp.length'] + epsilon)
                
            if 'tcp.payload_len' in X_out.columns and 'frame.len' in X_out.columns:
                X_out['payload_header_ratio'] = X_out['tcp.payload_len'] / (X_out['frame.len'] + epsilon)
                
            if 'frame.len' in X_out.columns and 'frame.time_delta' in X_out.columns:
                X_out['flow_rate'] = X_out['frame.len'] / (X_out['frame.time_delta'] + epsilon)
                
        # 3. Interactions
        if self.create_interactions:
            for col1, col2 in self.interaction_pairs_:
                if col1 in X_out.columns and col2 in X_out.columns:
                    new_col = f"{col1}_x_{col2}"
                    X_out[new_col] = X_out[col1] * X_out[col2]
                    
        # Replace any infs or NaNs generated during engineering
        new_cols = [c for c in self.engineered_feature_names_ if c in X_out.columns]
        if new_cols:
            X_out[new_cols] = X_out[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        logger.info(f"Engineered {len(new_cols)} features. Output shape: {X_out.shape}")
        return X_out
        
    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.feature_names_out_
