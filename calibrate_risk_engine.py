"""Calibrate and evaluate the transparent multi-signal risk engine."""

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.anomaly_detector import AnomalyDetector
from src.anomaly_features import AnomalyFeatureBuilder
from src.features import FraudFeatureBuilder
from src.model import FraudModel
from src.network_analyzer import NetworkAnalyzer
from src.rules import FraudRulesEngine
from src.schemas import TransactionInput
from train_anomaly_model import (
    DATASET,
    PAYSIM_EPOCH,
    add_past_only_history,
    build_feature_frame,
    load_data,
    resolve_data_path,
)

DEFAULT_CONFIG_PATH = Path("src/artifacts/risk_config.json")
RISK_ENGINE_VERSION = "2.0.0"
TARGET_RECALL = 0.95
MODEL_DECISION_THRESHOLD = 0.92
HIGH_RULE_FLOOR = 70.0
MEDIUM_REVIEW_QUANTILE = 0.95
RANDOM_STATE = 42
WEIGHT_CANDIDATES = [
    (0.65, 0.20, 0.10, 0.05),
    (0.60, 0.25, 0.10, 0.05),
    (0.60, 0.20, 0.15, 0.05),
    (0.60, 0.20, 0.10, 0.10),
    (0.55, 0.25, 0.10, 0.10),
    (0.55, 0.20, 0.15, 0.10),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def build_fraud_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    origin_count = data["orig_previous_transaction_count"].to_numpy()
    destination_count = data[
        "dest_previous_transaction_count"
    ].to_numpy()
    origin_average = np.divide(
        data["orig_previous_total_amount"].to_numpy(),
        origin_count,
        out=np.zeros(len(data), dtype=np.float64),
        where=origin_count > 0,
    )
    destination_average = np.divide(
        data["dest_previous_total_amount"].to_numpy(),
        destination_count,
        out=np.zeros(len(data), dtype=np.float64),
        where=destination_count > 0,
    )

    features = pd.DataFrame(
        {
            "step": data["step"],
            "amount": data["amount"],
            "oldbalanceOrg": data["oldbalanceOrg"],
            "oldbalanceDest": data["oldbalanceDest"],
            "hour_of_day": data["step"] % 24,
            "orig_previous_transaction_count": origin_count,
            "orig_previous_total_amount": data[
                "orig_previous_total_amount"
            ],
            "orig_previous_avg_amount": origin_average,
            "orig_time_since_previous": data[
                "orig_time_since_previous"
            ],
            "dest_previous_transaction_count": destination_count,
            "dest_previous_total_amount": data[
                "dest_previous_total_amount"
            ],
            "dest_previous_avg_amount": destination_average,
            "dest_time_since_previous": data[
                "dest_time_since_previous"
            ],
            "type_CASH_IN": (data["type"] == "CASH_IN").astype(float),
            "type_CASH_OUT": (data["type"] == "CASH_OUT").astype(float),
            "type_DEBIT": (data["type"] == "DEBIT").astype(float),
            "type_PAYMENT": (data["type"] == "PAYMENT").astype(float),
            "type_TRANSFER": (data["type"] == "TRANSFER").astype(float),
        },
        index=data.index,
    )
    return features.loc[
        :,
        FraudFeatureBuilder.FEATURE_NAMES,
    ].astype("float32")


def transaction_from_row(row: pd.Series) -> TransactionInput:
    source_index = int(row.get("original_index", row.name))
    return TransactionInput(
        transaction_id=f"paysim_{source_index:08d}",
        sender_id=str(row.get("nameOrig", f"sender_{source_index}")),
        receiver_id=str(
            row.get("nameDest", f"receiver_{source_index}")
        ),
        timestamp=PAYSIM_EPOCH + timedelta(hours=int(row["step"])),
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


def assert_batch_parity(
    data: pd.DataFrame,
    fraud_features: pd.DataFrame,
    anomaly_features: pd.DataFrame,
    high_rule_mask: np.ndarray,
) -> None:
    fraud_builder = FraudFeatureBuilder()
    anomaly_builder = AnomalyFeatureBuilder()
    rules_engine = FraudRulesEngine()
    sample = data.sample(n=min(20, len(data)), random_state=RANDOM_STATE)

    for index, row in sample.iterrows():
        transaction = transaction_from_row(row)
        runtime_fraud = fraud_builder.build(transaction)
        runtime_anomaly = anomaly_builder.build(transaction)
        batch_fraud = fraud_features.loc[index].to_numpy(dtype=np.float32)
        batch_anomaly = anomaly_features.loc[index].to_numpy(
            dtype=np.float32
        )

        expected_fraud = np.asarray(
            [
                runtime_fraud[name]
                for name in FraudFeatureBuilder.FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        expected_anomaly = np.asarray(
            [
                runtime_anomaly[name]
                for name in AnomalyFeatureBuilder.FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        if not np.allclose(batch_fraud, expected_fraud, rtol=1e-5):
            raise AssertionError(
                f"Fraud feature parity failed at row {index}."
            )
        if not np.allclose(batch_anomaly, expected_anomaly, rtol=1e-5):
            raise AssertionError(
                f"Anomaly feature parity failed at row {index}."
            )

        runtime_high_rule = any(
            result.severity == "HIGH"
            for result in rules_engine.evaluate(transaction)
        )
        if bool(high_rule_mask[data.index.get_loc(index)]) != runtime_high_rule:
            raise AssertionError(f"Rule parity failed at row {index}.")


def build_high_rule_mask(data: pd.DataFrame) -> np.ndarray:
    transaction_type = data["type"].astype("string")
    amount = data["amount"].to_numpy(dtype=np.float64)
    origin_balance = data["oldbalanceOrg"].to_numpy(dtype=np.float64)
    fraud_prone_type = transaction_type.isin(
        ["CASH_OUT", "TRANSFER"]
    ).to_numpy()
    allowed_difference = origin_balance * 0.01
    origin_drain = (
        fraud_prone_type
        & (origin_balance > 0)
        & (np.abs(amount - origin_balance) <= allowed_difference)
    )
    funded_transfer = (
        (transaction_type == "TRANSFER").to_numpy()
        & (amount >= 1_000_000.0)
        & (origin_balance >= amount)
    )
    return origin_drain | funded_transfer


def score_signals(
    data: pd.DataFrame,
    fraud_model: FraudModel,
    anomaly_detector: AnomalyDetector,
    network_analyzer: NetworkAnalyzer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fraud_features = build_fraud_feature_frame(data)
    fraud_matrix = xgb.DMatrix(
        fraud_features.to_numpy(dtype=np.float32),
        feature_names=FraudFeatureBuilder.FEATURE_NAMES,
    )
    model_scores = fraud_model.model.predict(fraud_matrix) * 100

    anomaly_features = build_feature_frame(data)
    raw_anomaly_scores = -anomaly_detector.model.score_samples(
        anomaly_features.to_numpy(dtype=np.float32)
    )
    anomaly_positions = np.searchsorted(
        anomaly_detector.calibration_scores,
        raw_anomaly_scores,
        side="left",
    )
    anomaly_scores = (
        anomaly_positions
        / len(anomaly_detector.calibration_scores)
        * 100
    )
    high_rule_mask = build_high_rule_mask(data)
    network_scores = np.asarray(
        [
            network_analyzer.analyze_and_record(
                transaction_from_row(row)
            ).network_score
            for _, row in data.iterrows()
        ],
        dtype=np.float32,
    )
    assert_batch_parity(
        data,
        fraud_features,
        anomaly_features,
        high_rule_mask,
    )
    return model_scores, anomaly_scores, high_rule_mask, network_scores


def combined_scores(
    model_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    high_rule_mask: np.ndarray,
    network_scores: np.ndarray,
    weights: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    model_weight, rule_weight, anomaly_weight, network_weight = weights
    rule_scores = high_rule_mask.astype(float) * 100
    weighted_scores = (
        model_scores * model_weight
        + rule_scores * rule_weight
        + anomaly_scores * anomaly_weight
        + network_scores * network_weight
    )
    risk_scores = np.where(
        high_rule_mask,
        np.maximum(weighted_scores, HIGH_RULE_FLOOR),
        weighted_scores,
    )
    return np.clip(risk_scores, 0, 100), weighted_scores


def select_high_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    model_weight: float,
) -> float | None:
    valid_thresholds = []
    maximum_threshold = min(
        HIGH_RULE_FLOOR,
        model_weight * MODEL_DECISION_THRESHOLD * 100,
    )
    for threshold in np.arange(40.0, maximum_threshold + 0.001, 0.5):
        recall = recall_score(
            labels,
            scores >= threshold,
            zero_division=0,
        )
        if recall >= TARGET_RECALL:
            valid_thresholds.append(float(threshold))
    return max(valid_thresholds, default=None)


def candidate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    high_threshold: float,
) -> dict[str, float]:
    predictions = scores >= high_threshold
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_configuration(
    labels: np.ndarray,
    model_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    high_rule_mask: np.ndarray,
    network_scores: np.ndarray,
) -> dict:
    candidates = []

    for weights in WEIGHT_CANDIDATES:
        scores, _ = combined_scores(
            model_scores,
            anomaly_scores,
            high_rule_mask,
            network_scores,
            weights,
        )
        high_threshold = select_high_threshold(
            labels,
            scores,
            weights[0],
        )
        if high_threshold is None:
            continue
        metrics = candidate_metrics(labels, scores, high_threshold)
        candidates.append(
            {
                "weights": weights,
                "risk_scores": scores,
                "high_threshold": high_threshold,
                "metrics": metrics,
            }
        )

    if not candidates:
        raise RuntimeError(
            "No risk configuration reached the validation recall target."
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["pr_auc"],
            candidate["metrics"]["precision"],
            -candidate["metrics"]["false_positive_rate"],
        ),
    )


def evaluate_split(
    split_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    medium_threshold: float,
    high_threshold: float,
) -> dict:
    metrics = candidate_metrics(labels, scores, high_threshold)
    levels = np.where(
        scores >= high_threshold,
        "HIGH",
        np.where(scores >= medium_threshold, "MEDIUM", "LOW"),
    )
    metrics["risk_levels"] = {
        "LOW": int(np.sum(levels == "LOW")),
        "MEDIUM": int(np.sum(levels == "MEDIUM")),
        "HIGH": int(np.sum(levels == "HIGH")),
    }

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(
        "Legitimate false-positive rate: "
        f"{metrics['false_positive_rate']:.4%}"
    )
    print(
        f"TN={metrics['tn']:,} FP={metrics['fp']:,} "
        f"FN={metrics['fn']:,} TP={metrics['tp']:,}"
    )
    print(f"Risk levels: {metrics['risk_levels']}")
    return metrics


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    print(f"Loading PaySim from {data_path}")
    data = add_past_only_history(load_data(data_path))
    validation = data.loc[data["step"].between(521, 631)]
    test = data.loc[data["step"] > 631]

    fraud_model = FraudModel()
    anomaly_detector = AnomalyDetector()
    network_analyzer = NetworkAnalyzer()
    network_history = data.loc[data["step"].between(497, 520)]
    network_analyzer.backfill(
        [
            transaction_from_row(row)
            for _, row in network_history.iterrows()
        ]
    )

    print(f"Scoring {len(validation):,} validation transactions")
    validation_signals = score_signals(
        validation,
        fraud_model,
        anomaly_detector,
        network_analyzer,
    )
    validation_labels = validation["isFraud"].to_numpy()
    selected = choose_configuration(
        validation_labels,
        *validation_signals,
    )
    (
        model_weight,
        rule_weight,
        anomaly_weight,
        network_weight,
    ) = selected["weights"]
    high_threshold = selected["high_threshold"]
    validation_scores = selected["risk_scores"]
    legitimate_scores = validation_scores[validation_labels == 0]
    medium_threshold = float(
        np.quantile(
            legitimate_scores,
            MEDIUM_REVIEW_QUANTILE,
            method="higher",
        )
    )
    medium_threshold = min(
        medium_threshold,
        float(np.nextafter(high_threshold, -np.inf)),
    )

    validation_metrics = evaluate_split(
        "VALIDATION",
        validation_labels,
        validation_scores,
        medium_threshold,
        high_threshold,
    )

    print(f"\nScoring {len(test):,} untouched test transactions")
    test_signals = score_signals(
        test,
        fraud_model,
        anomaly_detector,
        network_analyzer,
    )
    test_scores, _ = combined_scores(
        *test_signals,
        selected["weights"],
    )
    test_metrics = evaluate_split(
        "TEST",
        test["isFraud"].to_numpy(),
        test_scores,
        medium_threshold,
        high_threshold,
    )

    artifact = {
        "config": {
            "model_weight": model_weight,
            "rule_weight": rule_weight,
            "anomaly_weight": anomaly_weight,
            "network_weight": network_weight,
            "low_rule_score": 20.0,
            "medium_rule_score": 60.0,
            "high_rule_score": 100.0,
            "medium_threshold": medium_threshold,
            "high_threshold": high_threshold,
            "low_rule_floor": 0.0,
            "medium_rule_floor": max(30.0, medium_threshold),
            "high_rule_floor": HIGH_RULE_FLOOR,
            "version": RISK_ENGINE_VERSION,
        },
        "calibration": {
            "dataset": DATASET,
            "target_validation_recall": TARGET_RECALL,
            "preserved_model_decision_threshold": (
                MODEL_DECISION_THRESHOLD
            ),
            "medium_review_quantile": MEDIUM_REVIEW_QUANTILE,
            "validation_steps": [521, 631],
            "test_min_step": 632,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
    }
    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_path.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved risk configuration to {args.config_path}")


if __name__ == "__main__":
    main()
