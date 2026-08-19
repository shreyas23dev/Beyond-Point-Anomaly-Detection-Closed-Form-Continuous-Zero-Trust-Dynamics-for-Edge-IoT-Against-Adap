"""Test fixtures."""
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_data():
    """Create a small synthetic dataset mimicking Edge-IIoTset."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'tcp.port': np.random.randint(1, 65535, n_samples),
        'tcp.len': np.random.randint(0, 1500, n_samples),
        'udp.length': np.random.randint(0, 1500, n_samples),
        'frame.len': np.random.randint(40, 1500, n_samples),
        'frame.time_delta': np.random.exponential(0.1, n_samples),
        'tcp.payload_len': np.random.randint(0, 1460, n_samples),
        'ip.ttl': np.random.choice([64, 128, 255], n_samples),
        'http.request.method': np.random.choice(['GET', 'POST', '0'], n_samples), # 0 for non-http
        'Attack_type': np.random.choice(['Normal', 'DDoS_UDP', 'Port_Scanning'], n_samples, p=[0.7, 0.2, 0.1]),
    }
    
    df = pd.DataFrame(data)
    df['Attack_label'] = (df['Attack_type'] != 'Normal').astype(int)
    
    # Introduce some NaNs to test imputation
    df.loc[0:5, 'tcp.len'] = np.nan
    
    return df
