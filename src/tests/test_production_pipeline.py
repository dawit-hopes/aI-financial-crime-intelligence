import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_detector import evaluate_predictions
from scripts.train_fraud_model import (
    THRESHOLD_CANDIDATES,
    threshold_metrics,
)
from src.paysim_data import build_fraud_feature_frame, transaction_from_row


def test_threshold_candidates_include_preserved_operating_point():
    assert 0.92 in THRESHOLD_CANDIDATES


def test_threshold_metrics_shape():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.95, 0.8, 0.99])

    metrics = threshold_metrics(labels, probabilities, 0.92)

    assert metrics["threshold"] == 0.92
    assert {"precision", "recall", "f1", "false_positive_rate"} <= metrics.keys()


def test_evaluate_predictions_returns_bounded_scores():
    data = pd.DataFrame(
        {
            "type": ["TRANSFER", "PAYMENT"],
            "isFraud": [1, 0],
        }
    )
    metrics = evaluate_predictions(
        data,
        labels=np.asarray([1, 0]),
        risk_scores=np.asarray([90.0, 10.0], dtype=np.float32),
        model_scores=np.asarray([99.0, 1.0], dtype=np.float32),
        rule_scores=np.asarray([100.0, 0.0], dtype=np.float32),
        anomaly_scores=np.asarray([50.0, 20.0], dtype=np.float32),
        network_scores=np.asarray([0.0, 0.0], dtype=np.float32),
        medium_threshold=30.0,
        high_threshold=55.0,
    )

    assert 0.0 <= metrics["overall"]["precision"] <= 1.0
    assert metrics["overall"]["risk_levels"]["HIGH"] == 1
    assert metrics["overall"]["risk_levels"]["LOW"] == 1


def test_fraud_model_config_schema_when_present():
    config_path = Path("src/artifacts/fraud_model_config.json")
    if not config_path.exists():
        pytest.skip("fraud_model_config.json not generated yet")

    artifact = json.loads(config_path.read_text(encoding="utf-8"))
    assert artifact["decision_threshold"] == 0.92
    assert artifact["threshold_selection"] == "preserved_operating_point"
    assert artifact["threshold_selection_dataset"] == "validation"
    assert isinstance(artifact["threshold_candidates"], list)
