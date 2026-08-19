# Edge-IIoT Anomaly Detection

Machine Learning engine for detecting anomalies in Edge-IIoTset network traffic using an unsupervised Isolation Forest approach.

## Overview
This project implements Phase 1 of the Edge-IIoT Anomaly Detection system. It features a complete end-to-end pipeline including data downloading, preprocessing, feature engineering, hyperparameter tuning, model training, evaluation, SHAP explainability, and a FastAPI inference service.

## Setup

1. Create and activate a virtual environment (Python 3.11+):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Configure environment:
   Copy `.env.example` to `.env` and set your Kaggle credentials (or ensure `~/.kaggle/kaggle.json` is set up).

## Usage

### 1. Download Dataset
The dataset is downloaded from Kaggle using `src/data/download.py`. The training script handles this automatically if you've set up your Kaggle credentials.

### 2. Train the Model
Run the main orchestration script to execute the entire pipeline (data splitting, preprocessing, tuning, training, evaluation, and benchmarking):
```bash
python train.py
```

This will:
- Generate profiling reports in `data/profiling/`
- Track experiments using MLflow in `mlruns/`
- Save models to `artifacts/models/`
- Save evaluation plots and SHAP visualizations to `artifacts/plots/`
- Save benchmark and metadata results to `artifacts/metadata/` and `artifacts/results/`

### 3. Run the API
You can run the inference API locally:
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
Or using Docker:
```bash
docker-compose up --build
```

Access the API documentation at `http://localhost:8000/docs`.

## Testing
Run the unit tests:
```bash
pytest tests/
```

## Architecture

- **`config/`**: Centralized configuration management using dataclasses.
- **`src/data/`**: Data loading, profiling, and splitting (70/15/15 stratified).
- **`src/preprocessing/`**: Scikit-learn compatible transformers for cleaning, encoding, and scaling.
- **`src/features/`**: Feature engineering (logs, ratios, interactions) and selection (variance, correlation).
- **`src/model/`**: Isolation Forest wrapper, threshold calibration, CV, and evaluation.
- **`src/explainability/`**: SHAP TreeExplainer integration.
- **`src/tracking/`**: MLflow integration.
- **`src/benchmarking/`**: Latency, throughput, and memory footprint profiling.
- **`src/api/`**: FastAPI service for real-time inference.
