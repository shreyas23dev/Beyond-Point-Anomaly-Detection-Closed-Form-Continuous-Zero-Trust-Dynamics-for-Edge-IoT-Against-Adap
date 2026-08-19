"""Feature selection transformer."""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Transformer for selecting the best features."""
    
    def __init__(self, variance_threshold: float = 0.01, correlation_threshold: float = 0.95, max_features: int | None = None):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        
        self.selected_features_: list[str] = []
        self.dropped_low_variance_: list[str] = []
        self.dropped_high_correlation_: dict[str, str] = {}
        self.feature_variances_: dict[str, float] = {}
        
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureSelector':
        """Identify features to keep based on variance and correlation."""
        logger.info("Fitting FeatureSelector...")
        initial_features = list(X.columns)
        
        # We only apply these filters to numeric features.
        # Categorical features that were OrdinalEncoded are numeric now.
        numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        
        X_num = X[numeric_cols]
        
        # 1. Variance Threshold
        variances = X_num.var()
        self.feature_variances_ = variances.to_dict()
        
        low_variance_mask = variances < self.variance_threshold
        self.dropped_low_variance_ = variances[low_variance_mask].index.tolist()
        
        logger.info(f"Dropping {len(self.dropped_low_variance_)} features due to low variance (< {self.variance_threshold}).")
        
        # Keep features passing variance threshold
        features_to_check_corr = [c for c in numeric_cols if c not in self.dropped_low_variance_]
        
        # 2. Correlation Filter
        self.dropped_high_correlation_ = {}
        if len(features_to_check_corr) > 1:
            # Compute correlation matrix on remaining numeric features
            corr_matrix = X_num[features_to_check_corr].corr().abs()
            
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            to_drop_corr = set()
            for col in upper.columns:
                high_corr_indices = upper[col][upper[col] > self.correlation_threshold].index.tolist()
                for high_corr_col in high_corr_indices:
                    if col not in to_drop_corr and high_corr_col not in to_drop_corr:
                        # Drop the one with lower variance to break the tie
                        var1 = variances[col]
                        var2 = variances[high_corr_col]
                        if var1 < var2:
                            to_drop_corr.add(col)
                            self.dropped_high_correlation_[col] = high_corr_col
                        else:
                            to_drop_corr.add(high_corr_col)
                            self.dropped_high_correlation_[high_corr_col] = col
                            
            logger.info(f"Dropping {len(self.dropped_high_correlation_)} features due to high correlation (> {self.correlation_threshold}).")
            
        # 3. Final Selection
        all_dropped = set(self.dropped_low_variance_) | set(self.dropped_high_correlation_.keys())
        
        self.selected_features_ = [f for f in initial_features if f not in all_dropped]
        
        # If max_features is set and we have too many, keep the ones with highest variance
        if self.max_features is not None and len(self.selected_features_) > self.max_features:
            logger.info(f"Reducing features from {len(self.selected_features_)} to max_features={self.max_features}.")
            # Separate numeric and non-numeric in selected
            sel_num = [f for f in self.selected_features_ if f in numeric_cols]
            sel_non_num = [f for f in self.selected_features_ if f in non_numeric_cols]
            
            # Sort numeric by variance descending
            sel_num_sorted = sorted(sel_num, key=lambda x: variances[x], reverse=True)
            
            # Keep all non-numeric (if any), then pad with top numeric
            keep_num = self.max_features - len(sel_non_num)
            if keep_num > 0:
                self.selected_features_ = sel_non_num + sel_num_sorted[:keep_num]
            else:
                self.selected_features_ = sel_non_num[:self.max_features]
                
        logger.info(f"Final selected features: {len(self.selected_features_)} out of {len(initial_features)}")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep only selected features."""
        logger.info(f"Transforming data with FeatureSelector. Keeping {len(self.selected_features_)} features.")
        # Only select features that actually exist in X (important for inference if some are missing)
        cols_to_keep = [c for c in self.selected_features_ if c in X.columns]
        return X[cols_to_keep].copy()
        
    def get_selection_report(self) -> dict:
        """Return a summary of feature selection."""
        return {
            "n_initial_features": len(self.selected_features_) + len(self.dropped_low_variance_) + len(self.dropped_high_correlation_),
            "n_selected_features": len(self.selected_features_),
            "n_dropped_low_variance": len(self.dropped_low_variance_),
            "n_dropped_high_correlation": len(self.dropped_high_correlation_),
            "dropped_low_variance": self.dropped_low_variance_,
            "dropped_high_correlation": self.dropped_high_correlation_
        }
        
    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.selected_features_
