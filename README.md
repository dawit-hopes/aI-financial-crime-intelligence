# Fraud Detection ML

Multi-signal fraud detection prototype for direct backend integration via the `FraudDetector` Python class.

## Setup

```bash
uv sync
```

## Backend integration

The backend calls `FraudDetector.predict()` directly. See [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md) for the contract, lifecycle, and worker requirements.

Network-specific behavior is documented in [NETWORK_INTELLIGENCE.md](NETWORK_INTELLIGENCE.md).

## Train, calibrate, and evaluate

Run in order against PaySim:

```bash
uv run python -m scripts.train_fraud_model --data-path /path/to/paysim.csv
uv run python -m scripts.train_anomaly_model --data-path /path/to/paysim.csv
uv run python -m scripts.calibrate_network_analyzer --data-path /path/to/paysim.csv
uv run python -m scripts.calibrate_risk_engine --data-path /path/to/paysim.csv
uv run python -m scripts.evaluate_detector --data-path /path/to/paysim.csv
```

If `--data-path` is omitted, scripts download PaySim via `kagglehub`.

Artifacts are written to `src/artifacts/`.

## Demo runners

```bash
uv run python -m scripts.run_detector_demo
uv run python -m scripts.run_detector_response --example 1
```

## Tests

```bash
uv run pytest
```

## Signals

The detector combines four signals into a 0–100 risk score:

- XGBoost supervised model
- Rule-based detection
- Isolation Forest anomaly detection
- NetworkX graph intelligence

Final fraud decisions come from the calibrated risk engine in `src/risk_engine.py`.
