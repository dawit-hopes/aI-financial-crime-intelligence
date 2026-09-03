import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.anomaly_detector import (
    ANOMALY_MODEL_PATH,
    AnomalyDetector,
)
from src.anomaly_features import AnomalyFeatureBuilder
from src.schemas import TransactionInput, TransactionType
from train_anomaly_model import build_feature_frame

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fraud_examples.csv"


def make_transaction(
    *,
    step: int = 24,
    transaction_type: TransactionType = "TRANSFER",
    amount: float = 500.0,
    origin_balance: float = 1_000.0,
    destination_balance: float = 100.0,
    origin_count: int = 2,
    origin_total: float = 600.0,
    origin_elapsed: float = 3.0,
    destination_count: int = 4,
    destination_total: float = 800.0,
    destination_elapsed: float = 2.0,
) -> TransactionInput:
    return TransactionInput(
        step=step,
        type=transaction_type,
        amount=amount,
        oldbalanceOrg=origin_balance,
        oldbalanceDest=destination_balance,
        orig_previous_transaction_count=origin_count,
        orig_previous_total_amount=origin_total,
        orig_time_since_previous=origin_elapsed,
        dest_previous_transaction_count=destination_count,
        dest_previous_total_amount=destination_total,
        dest_time_since_previous=destination_elapsed,
    )


def load_fixture_transactions() -> list[TransactionInput]:
    transactions = []

    with FIXTURE_PATH.open(newline="") as fixture:
        for row in csv.DictReader(fixture):
            transactions.append(
                TransactionInput(
                    step=int(row["step"]),
                    type=row["type"],
                    amount=float(row["amount"]),
                    oldbalanceOrg=float(row["oldbalanceOrg"]),
                    oldbalanceDest=float(row["oldbalanceDest"]),
                    orig_previous_transaction_count=int(
                        row["orig_previous_transaction_count"]
                    ),
                    orig_previous_total_amount=float(
                        row["orig_previous_total_amount"]
                    ),
                    orig_time_since_previous=float(
                        row["orig_time_since_previous"]
                    ),
                    dest_previous_transaction_count=int(
                        row["dest_previous_transaction_count"]
                    ),
                    dest_previous_total_amount=float(
                        row["dest_previous_total_amount"]
                    ),
                    dest_time_since_previous=float(
                        row["dest_time_since_previous"]
                    ),
                )
            )

    return transactions


def write_test_artifact(path: Path, **overrides) -> None:
    feature_count = len(AnomalyFeatureBuilder.FEATURE_NAMES)
    training_data = np.asarray(
        [
            np.linspace(0.0, 1.0, feature_count),
            np.linspace(0.2, 1.2, feature_count),
            np.linspace(0.4, 1.4, feature_count),
            np.linspace(0.6, 1.6, feature_count),
        ],
        dtype=np.float32,
    )
    model = IsolationForest(
        n_estimators=5,
        max_samples=4,
        random_state=42,
    ).fit(training_data)
    calibration_scores = np.sort(
        -model.score_samples(training_data)
    ).astype(np.float32)
    artifact = {
        "model": model,
        "feature_names": AnomalyFeatureBuilder.FEATURE_NAMES,
        "calibration_scores": calibration_scores,
        "raw_threshold": float(calibration_scores[-1]),
        "threshold": 0.75,
        "model_version": "test",
    }
    artifact.update(overrides)
    joblib.dump(artifact, path)


def test_anomaly_feature_builder_returns_ordered_finite_features():
    builder = AnomalyFeatureBuilder()

    features = builder.build(make_transaction())

    assert list(features) == builder.FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())


def test_anomaly_features_exclude_absolute_step():
    builder = AnomalyFeatureBuilder()

    first = builder.build(make_transaction(step=24))
    later = builder.build(make_transaction(step=240))

    assert first == pytest.approx(later)
    assert "step" not in first


def test_anomaly_features_encode_missing_history():
    features = AnomalyFeatureBuilder().build(
        make_transaction(
            origin_count=0,
            origin_total=0.0,
            origin_elapsed=-1.0,
            destination_count=0,
            destination_total=0.0,
            destination_elapsed=-1.0,
        )
    )

    assert features["orig_history_missing"] == 1.0
    assert features["dest_history_missing"] == 1.0
    assert features["log_orig_time_since_previous"] == 0.0
    assert features["log_dest_time_since_previous"] == 0.0


def test_anomaly_features_log_transform_heavy_tailed_values():
    features = AnomalyFeatureBuilder().build(
        make_transaction(amount=1_000_000.0)
    )

    assert features["log_amount"] == pytest.approx(
        np.log1p(1_000_000.0)
    )
    assert features["log_amount"] < 1_000_000.0


def test_vectorized_training_features_match_runtime_features():
    transaction = make_transaction()
    training_row = pd.DataFrame(
        [
            {
                "step": transaction.step,
                "type": transaction.type,
                "amount": transaction.amount,
                "oldbalanceOrg": transaction.oldbalanceOrg,
                "oldbalanceDest": transaction.oldbalanceDest,
                "orig_previous_transaction_count": (
                    transaction.orig_previous_transaction_count
                ),
                "orig_previous_total_amount": (
                    transaction.orig_previous_total_amount
                ),
                "orig_time_since_previous": (
                    transaction.orig_time_since_previous
                ),
                "dest_previous_transaction_count": (
                    transaction.dest_previous_transaction_count
                ),
                "dest_previous_total_amount": (
                    transaction.dest_previous_total_amount
                ),
                "dest_time_since_previous": (
                    transaction.dest_time_since_previous
                ),
            }
        ]
    )
    batch_features = build_feature_frame(training_row).iloc[0]
    runtime_features = AnomalyFeatureBuilder().build(transaction)

    assert batch_features.to_numpy() == pytest.approx(
        [runtime_features[name] for name in batch_features.index],
        rel=1e-5,
    )


def test_anomaly_detector_requires_existing_artifact(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="not found"):
        AnomalyDetector(missing_path)


def test_anomaly_detector_rejects_non_dictionary_artifact(tmp_path):
    artifact_path = tmp_path / "invalid.joblib"
    joblib.dump(["not", "a", "dictionary"], artifact_path)

    with pytest.raises(ValueError, match="dictionary"):
        AnomalyDetector(artifact_path)


def test_anomaly_detector_rejects_missing_artifact_keys(tmp_path):
    artifact_path = tmp_path / "missing-keys.joblib"
    joblib.dump({}, artifact_path)

    with pytest.raises(ValueError, match="missing required keys"):
        AnomalyDetector(artifact_path)


def test_anomaly_detector_rejects_feature_order_mismatch(tmp_path):
    artifact_path = tmp_path / "wrong-features.joblib"
    write_test_artifact(artifact_path, feature_names=["wrong"])

    with pytest.raises(ValueError, match="feature order"):
        AnomalyDetector(artifact_path)


def test_anomaly_detector_rejects_unsorted_calibration(tmp_path):
    artifact_path = tmp_path / "unsorted.joblib"
    write_test_artifact(
        artifact_path,
        calibration_scores=np.asarray([0.5, 0.4], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="finite, sorted"):
        AnomalyDetector(artifact_path)


def test_anomaly_detector_scores_are_deterministic(tmp_path):
    artifact_path = tmp_path / "valid.joblib"
    write_test_artifact(artifact_path)
    detector = AnomalyDetector(artifact_path)
    transaction = make_transaction()

    first = detector.predict(transaction)
    second = detector.predict(transaction)

    assert first == second
    assert 0.0 <= first.anomaly_score <= 1.0
    assert first.is_anomaly == (
        first.anomaly_score >= first.threshold
    )
    assert np.isfinite(first.raw_anomaly_score)


def test_packaged_anomaly_model_scores_all_fixture_transactions():
    assert ANOMALY_MODEL_PATH.exists()
    detector = AnomalyDetector()

    results = [
        detector.predict(transaction)
        for transaction in load_fixture_transactions()
    ]

    assert len(results) == 20
    assert all(0.0 <= result.anomaly_score <= 1.0 for result in results)
    assert all(
        result.is_anomaly
        == (result.anomaly_score >= result.threshold)
        for result in results
    )
