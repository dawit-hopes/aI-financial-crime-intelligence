"""Calibrate transparent NetworkX thresholds on chronological PaySim data."""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.network_analyzer import NetworkAnalyzer
from src.schemas import NetworkConfig, TransactionInput
from train_anomaly_model import DATASET, PAYSIM_EPOCH, resolve_data_path

DEFAULT_CONFIG_PATH = Path("src/artifacts/network_config.json")
TRAIN_START_STEP = 497
TRAIN_END_STEP = 520
VALIDATION_END_STEP = 631


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def load_network_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        usecols=[
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "isFraud",
        ],
    )
    data["source_row"] = data.index
    return data.loc[data["step"] >= TRAIN_START_STEP].copy()


def calibrated_config(training: pd.DataFrame) -> NetworkConfig:
    fan_out = training.groupby("nameOrig")["nameDest"].nunique()
    fan_in = training.groupby("nameDest")["nameOrig"].nunique()
    repeated = training.groupby(["nameOrig", "nameDest"]).size()
    volume = training.groupby("nameOrig")["amount"].sum()

    return NetworkConfig(
        fan_out_threshold=max(3, int(np.quantile(fan_out, 0.99))),
        fan_in_threshold=max(3, int(np.quantile(fan_in, 0.99))),
        repeated_pair_threshold=max(
            3,
            int(np.quantile(repeated, 0.99)),
        ),
        sender_volume_threshold=max(
            100_000.0,
            float(np.quantile(volume, 0.99)),
        ),
    )


def transaction_from_row(row: pd.Series) -> TransactionInput:
    return TransactionInput(
        transaction_id=f"paysim_{int(row['source_row']):08d}",
        sender_id=str(row["nameOrig"]),
        receiver_id=str(row["nameDest"]),
        timestamp=PAYSIM_EPOCH + timedelta(hours=int(row["step"])),
        step=int(row["step"]),
        type=row["type"],
        amount=float(row["amount"]),
        oldbalanceOrg=0.0,
        oldbalanceDest=0.0,
        orig_previous_transaction_count=0,
        orig_previous_total_amount=0.0,
        orig_time_since_previous=-1.0,
        dest_previous_transaction_count=0,
        dest_previous_total_amount=0.0,
        dest_time_since_previous=-1.0,
    )


def replay(
    analyzer: NetworkAnalyzer,
    data: pd.DataFrame,
) -> np.ndarray:
    return np.asarray(
        [
            analyzer.analyze_and_record(
                transaction_from_row(row)
            ).network_score
            for _, row in data.iterrows()
        ],
        dtype=np.float32,
    )


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "mean_score": float(np.mean(scores)),
        "p99_score": float(np.quantile(scores, 0.99)),
        "positive_score_rate": float(np.mean(scores > 0)),
    }


def main() -> None:
    args = parse_args()
    path = resolve_data_path(args.data_path)
    data = load_network_data(path)
    training = data.loc[data["step"] <= TRAIN_END_STEP]
    validation = data.loc[
        data["step"].between(TRAIN_END_STEP + 1, VALIDATION_END_STEP)
    ]
    test = data.loc[data["step"] > VALIDATION_END_STEP]
    config = calibrated_config(training)
    analyzer = NetworkAnalyzer(config=config)

    print(f"Backfilling {len(training):,} training-tail transactions")
    analyzer.backfill(
        [
            transaction_from_row(row)
            for _, row in training.iterrows()
        ]
    )
    print(f"Scoring {len(validation):,} validation transactions")
    validation_scores = replay(analyzer, validation)
    validation_metrics = metrics(
        validation["isFraud"].to_numpy(),
        validation_scores,
    )
    print(f"Validation metrics: {validation_metrics}")

    print(f"Scoring {len(test):,} test transactions")
    test_scores = replay(analyzer, test)
    test_metrics = metrics(test["isFraud"].to_numpy(), test_scores)
    print(f"Test metrics: {test_metrics}")

    artifact = {
        "config": config.model_dump(),
        "calibration": {
            "dataset": DATASET,
            "training_steps": [TRAIN_START_STEP, TRAIN_END_STEP],
            "validation_steps": [
                TRAIN_END_STEP + 1,
                VALIDATION_END_STEP,
            ],
            "test_min_step": VALIDATION_END_STEP + 1,
            "labels_used_for_thresholds": False,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
    }
    args.config_path.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved network configuration to {args.config_path}")


if __name__ == "__main__":
    main()
