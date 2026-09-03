"""Calibrate transparent NetworkX thresholds on chronological PaySim data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.network_analyzer import NetworkAnalyzer
from src.paysim_data import (
    DATASET,
    NETWORK_BACKFILL_MAX_STEP,
    NETWORK_BACKFILL_MIN_STEP,
    VALIDATION_MAX_STEP,
    resolve_data_path,
    transaction_from_row,
)
from src.schemas import NetworkConfig

DEFAULT_CONFIG_PATH = Path("src/artifacts/network_config.json")


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
    return data.loc[data["step"] >= NETWORK_BACKFILL_MIN_STEP].copy()


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
    training = data.loc[data["step"] <= NETWORK_BACKFILL_MAX_STEP]
    validation = data.loc[
        data["step"].between(
            NETWORK_BACKFILL_MAX_STEP + 1,
            VALIDATION_MAX_STEP,
        )
    ]
    test = data.loc[data["step"] > VALIDATION_MAX_STEP]
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
            "training_steps": [
                NETWORK_BACKFILL_MIN_STEP,
                NETWORK_BACKFILL_MAX_STEP,
            ],
            "validation_steps": [
                NETWORK_BACKFILL_MAX_STEP + 1,
                VALIDATION_MAX_STEP,
            ],
            "test_min_step": VALIDATION_MAX_STEP + 1,
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
