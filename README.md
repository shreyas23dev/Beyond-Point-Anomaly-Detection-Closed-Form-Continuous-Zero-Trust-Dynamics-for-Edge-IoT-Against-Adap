# Continuous Zero-Trust Calibration for Edge-IoT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker: Containerized](https://img.shields.io/badge/Docker-Microservices-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

> **Official Replication Repository** for research on *Continuous Zero-Trust Calibration and Asymmetric Slow-Burn Attack Detection for Edge-IoT*.

---

## 📌 Repository Overview

This repository provides the reference source code, microservice deployment configurations, and reproduction scripts for an end-to-end Dynamic Zero-Trust behavioral calibration pipeline designed for edge infrastructure.

### Core Modules:
* **`dynamic-trust-engine/`**: Implements continuous trust calibration, asymmetric evidence accumulation, and policy-driven recovery dynamics.
* **`edge-iiot-anomaly-detection/`**: Unsupervised Isolation Forest anomaly detection pipeline trained on Edge-IIoTset network traffic.
* **`zero-trust-engine/`**: Zero-Trust Policy Decision Point (PDP) rule execution microservice.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/shreyas23dev/Continuous-Zero-Trust-Calibration-for-Edge-IoT.git
cd Continuous-Zero-Trust-Calibration-for-Edge-IoT

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r edge-iiot-anomaly-detection/requirements.txt
pip install -r dynamic-trust-engine/requirements.txt
```

### 2. Dataset Preparation
This project evaluates on the public **Edge-IIoTset** benchmark dataset:
- **Download:** [IEEE Dataport Edge-IIoTset](https://doi.org/10.21227/mbc1-1h68) (or Kaggle mirror).
- Place `DNN-EdgeIIoT-dataset.csv` into `edge-iiot-anomaly-detection/data/raw/`.

### 3. Run Experiments
```bash
# Train ML baseline and run evaluation pipeline
python edge-iiot-anomaly-detection/train.py

# Run Trust Engine unit and benchmark tests
pytest dynamic-trust-engine/tests/
```

### 4. Launch Microservices (Optional)
To run the complete containerized stack locally via Docker Compose:
```bash
cd zero-trust-engine/docker
docker compose up -d
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
