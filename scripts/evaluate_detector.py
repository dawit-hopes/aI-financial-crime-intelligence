"""Evaluate the full FraudDetector on chronological PaySim splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from scripts.calibrate_risk_engine import combined_scores, score_signals
from src.anomaly_detector import AnomalyDetector
from src.fraud_detector import FraudDetector
from src.model import FraudModel
from src.network_analyzer import NetworkAnalyzer
from src.paysim_data import (
    DATASET,
    NETWORK_BACKFILL_MAX_STEP,
    NETWORK_BACKFILL_MIN_STEP,
    TEST_MIN_STEP,
    VALIDATION_MAX_STEP,
    VALIDATION_MIN_STEP,
    add_past_only_history,
    load_data,
    resolve_data_path,
    transaction_from_row,
)
from src.risk_engine import RiskEngine

DEFAULT_REPORT_PATH = Path("src/artifacts/evaluation_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    return parser.parse_args()


def load_artifact_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    artifact_paths = {
        "fraud_model_config": Path("src/artifacts/fraud_model_config.json"),
        "risk_config": Path("src/artifacts/risk_config.json"),
        "network_config": Path("src/artifacts/network_config.json"),
    }
    for name, path in artifact_paths.items():
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if name == "fraud_model_config":
                versions[name] = artifact.get("model_version", "unknown")
            elif "config" in artifact:
                versions[name] = artifact["config"].get("version", "unknown")
    return versions


def evaluate_predictions(
    data: pd.DataFrame,
    labels: np.ndarray,
    risk_scores: np.ndarray,
    model_scores: np.ndarray,
    rule_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    network_scores: np.ndarray,
    medium_threshold: float,
    high_threshold: float,
) -> dict:
    predictions = (risk_scores >= high_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    risk_levels = np.where(
        risk_scores >= high_threshold,
        "HIGH",
        np.where(risk_scores >= medium_threshold, "MEDIUM", "LOW"),
    )

    overall = {
        "pr_auc": float(average_precision_score(labels, risk_scores)),
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
        "risk_levels": {
            level: int(np.sum(risk_levels == level))
            for level in ("LOW", "MEDIUM", "HIGH")
        },
        "signal_summary": {
            "mean_model_probability": float(np.mean(model_scores) / 100),
            "rule_trigger_rate": float(np.mean(rule_scores > 0)),
            "mean_anomaly_score": float(np.mean(anomaly_scores) / 100),
            "network_positive_rate": float(np.mean(network_scores > 0)),
        },
    }

    by_type: dict[str, dict] = {}
    reset_data = data.reset_index(drop=True)
    for transaction_type, group in reset_data.groupby(
        "type",
        observed=True,
    ):
        positions = group.index.to_numpy()
        type_labels = group["isFraud"].to_numpy()
        type_predictions = predictions[positions]
        type_tn, type_fp, type_fn, type_tp = confusion_matrix(
            type_labels,
            type_predictions,
            labels=[0, 1],
        ).ravel()
        by_type[str(transaction_type)] = {
            "count": int(len(group)),
            "fraud_count": int(type_labels.sum()),
            "precision": float(
                precision_score(
                    type_labels,
                    type_predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    type_labels,
                    type_predictions,
                    zero_division=0,
                )
            ),
            "false_positive_rate": float(
                type_fp / (type_fp + type_tn) if (type_fp + type_tn) else 0.0
            ),
        }

    return {
        "overall": overall,
        "by_transaction_type": by_type,
    }


def score_split(
    data: pd.DataFrame,
    fraud_model: FraudModel,
    anomaly_detector: AnomalyDetector,
    network_analyzer: NetworkAnalyzer,
    risk_engine: RiskEngine,
) -> dict[str, np.ndarray]:
    (
        model_scores,
        anomaly_scores,
        high_rule_mask,
        network_scores,
    ) = score_signals(
        data,
        fraud_model,
        anomaly_detector,
        network_analyzer,
    )
    weights = (
        risk_engine.config.model_weight,
        risk_engine.config.rule_weight,
        risk_engine.config.anomaly_weight,
        risk_engine.config.network_weight,
    )
    risk_scores, _ = combined_scores(
        model_scores,
        anomaly_scores,
        high_rule_mask,
        network_scores,
        weights,
    )

    return {
        "risk_scores": np.asarray(risk_scores, dtype=np.float32),
        "model_scores": np.asarray(model_scores, dtype=np.float32),
        "rule_scores": high_rule_mask.astype(np.float32) * 100,
        "anomaly_scores": np.asarray(anomaly_scores, dtype=np.float32),
        "network_scores": np.asarray(network_scores, dtype=np.float32),
    }


def assert_runtime_parity(
    data: pd.DataFrame,
    network_history: pd.DataFrame,
    batch_scores: np.ndarray,
) -> None:
    sample = data.iloc[: min(20, len(data))]
    analyzer = NetworkAnalyzer()
    analyzer.backfill(
        [
            transaction_from_row(row)
            for _, row in network_history.iterrows()
        ]
    )
    detector = FraudDetector(network_analyzer=analyzer)
    runtime_scores = np.asarray(
        [
            detector.predict(transaction_from_row(row)).risk_score
            for _, row in sample.iterrows()
        ]
    )
    if not np.allclose(runtime_scores, batch_scores[: len(sample)], rtol=1e-5):
        raise AssertionError(
            "Vectorized evaluation does not match FraudDetector runtime."
        )


def print_split_metrics(split_name: str, metrics: dict) -> None:
    overall = metrics["overall"]
    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"PR-AUC: {overall['pr_auc']:.4f}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall: {overall['recall']:.4f}")
    print(f"F1: {overall['f1']:.4f}")
    print(
        "Legitimate false-positive rate: "
        f"{overall['false_positive_rate']:.4%}"
    )
    print(
        f"TN={overall['tn']:,} FP={overall['fp']:,} "
        f"FN={overall['fn']:,} TP={overall['tp']:,}"
    )
    print(f"Risk levels: {overall['risk_levels']}")
    print(f"Signal summary: {overall['signal_summary']}")


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    print(f"Loading PaySim from {data_path}")
    data = add_past_only_history(load_data(data_path))

    validation = data.loc[
        data["step"].between(VALIDATION_MIN_STEP, VALIDATION_MAX_STEP)
    ]
    test = data.loc[data["step"] >= TEST_MIN_STEP]
    network_history = data.loc[
        data["step"].between(
            NETWORK_BACKFILL_MIN_STEP,
            NETWORK_BACKFILL_MAX_STEP,
        )
    ]

    fraud_model = FraudModel()
    anomaly_detector = AnomalyDetector()
    network_analyzer = NetworkAnalyzer()
    risk_engine = RiskEngine()
    high_threshold = risk_engine.config.high_threshold
    medium_threshold = risk_engine.config.medium_threshold

    print(
        f"Backfilling {len(network_history):,} network history transactions"
    )
    network_analyzer.backfill(
        [
            transaction_from_row(row)
            for _, row in network_history.iterrows()
        ]
    )

    print(f"Evaluating {len(validation):,} validation transactions")
    validation_predictions = score_split(
        validation,
        fraud_model,
        anomaly_detector,
        network_analyzer,
        risk_engine,
    )
    assert_runtime_parity(
        validation,
        network_history,
        validation_predictions["risk_scores"],
    )
    validation_metrics = evaluate_predictions(
        validation,
        validation["isFraud"].to_numpy(),
        **validation_predictions,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    print_split_metrics("VALIDATION", validation_metrics)

    print(f"\nEvaluating {len(test):,} untouched test transactions")
    test_predictions = score_split(
        test,
        fraud_model,
        anomaly_detector,
        network_analyzer,
        risk_engine,
    )
    test_metrics = evaluate_predictions(
        test,
        test["isFraud"].to_numpy(),
        **test_predictions,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    print_split_metrics("TEST", test_metrics)

    report = {
        "dataset": DATASET,
        "splits": {
            "validation_steps": [VALIDATION_MIN_STEP, VALIDATION_MAX_STEP],
            "test_min_step": TEST_MIN_STEP,
            "network_backfill_steps": [
                NETWORK_BACKFILL_MIN_STEP,
                NETWORK_BACKFILL_MAX_STEP,
            ],
        },
        "artifact_versions": load_artifact_versions(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved evaluation report to {args.report_path}")


if __name__ == "__main__":
    main()
