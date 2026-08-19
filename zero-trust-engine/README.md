# Zero Trust Decision Engine
Phase 3 of the Anomaly Detection Pipeline.

This standalone microservice consumes inputs from the Dynamic Trust Engine and evaluates them against deterministic policies (ALLOW/VERIFY/BLOCK) using a purely deterministic evaluation logic.

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

## Running
```bash
uvicorn src.zta_api:app --reload
```

## Documentation
See `docs/zta_report.md` for full implementation details.
