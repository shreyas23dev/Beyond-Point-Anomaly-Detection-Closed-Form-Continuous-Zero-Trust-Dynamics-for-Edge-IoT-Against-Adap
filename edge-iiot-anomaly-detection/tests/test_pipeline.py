"""Tests for the preprocessing and modeling pipeline."""
import numpy as np
import pandas as pd

from src.model.isolation_forest import AnomalyDetector
from src.preprocessing.pipeline import PreprocessingPipeline

def test_preprocessing_pipeline_fit_transform(sample_data):
    """Test that pipeline fits and transforms without errors and handles NaNs."""
    pipeline = PreprocessingPipeline(
        columns_to_drop=['tcp.port'],
        target_columns=['Attack_type', 'Attack_label']
    )
    
    # Check shape before
    assert sample_data.shape == (100, 10)
    assert sample_data['tcp.len'].isna().sum() == 6
    
    # Fit transform
    X_out = pipeline.fit_transform(sample_data)
    
    # Target columns and dropped columns should be gone
    assert 'Attack_type' not in X_out.columns
    assert 'Attack_label' not in X_out.columns
    assert 'tcp.port' not in X_out.columns
    
    # NaNs should be imputed
    assert X_out['tcp.len'].isna().sum() == 0
    
    # Categorical columns should be encoded (numeric)
    assert pd.api.types.is_numeric_dtype(X_out['http.request.method'])
    
def test_isolation_forest_model(sample_data):
    """Test model training and scoring."""
    pipeline = PreprocessingPipeline(
        columns_to_drop=['tcp.port'],
        target_columns=['Attack_type', 'Attack_label']
    )
    
    X_train = sample_data[sample_data['Attack_label'] == 0]
    X_test = sample_data
    
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    model = AnomalyDetector(n_estimators=10, random_state=42)
    model.fit(X_train_processed)
    
    # Test normalization bounds
    assert model.score_min_ is not None
    assert model.score_max_ is not None
    
    # Test scoring
    scores = model.score_samples(X_test_processed)
    assert len(scores) == len(X_test)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    
    # Test predict without threshold should fail
    import pytest
    with pytest.raises(RuntimeError):
        model.predict(X_test_processed)
        
    # Set threshold and test predict
    model.set_threshold(0.5)
    preds = model.predict(X_test_processed)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})
    
    # Test predict_proba
    probas = model.predict_proba(X_test_processed)
    assert len(probas) == len(X_test)
    assert "confidence" in probas[0]
    assert "is_anomaly" in probas[0]
