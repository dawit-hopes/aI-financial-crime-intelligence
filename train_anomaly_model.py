"""Train and evaluate the standalone PaySim Isolation Forest model."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.anomaly_features import (
    MAX_RATIO,
    AnomalyFeatureBuilder,
)
from src.schemas import TransactionInput

DATASET = "ealaxi/paysim1"
DEFAULT_ARTIFACT_PATH = Path("src/artifacts/anomaly_model.joblib")
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
PAYSIM_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
MAX_TRAIN_SAMPLES = 250_000
MAX_FALSE_POSITIVE_RATE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional path to the PaySim CSV.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=MAX_TRAIN_SAMPLES,
    )
    return parser.parse_args()


def resolve_data_path(data_path: Path | None) -> Path:
    if data_path is not None:
        if not data_path.exists():
            raise FileNotFoundError(f"PaySim dataset not found: {data_path}")
        return data_path

    dataset_directory = Path(kagglehub.dataset_download(DATASET))
    csv_paths = sorted(dataset_directory.glob("*.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(
            "Expected exactly one PaySim CSV in "
            f"{dataset_directory}, found {len(csv_paths)}."
        )
    return csv_paths[0]


def load_data(data_path: Path) -> pd.DataFrame:
    columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "nameDest",
        "oldbalanceDest",
        "isFraud",
    ]
    data = pd.read_csv(
        data_path,
        usecols=columns,
        dtype={
            "step": "int16",
            "type": "category",
            "amount": "float32",
            "oldbalanceOrg": "float32",
            "oldbalanceDest": "float32",
            "isFraud": "int8",
        },
    )
    data["original_index"] = np.arange(len(data), dtype=np.int32)
    return data.sort_values(
        ["step", "original_index"],
        kind="stable",
    ).reset_index(drop=True)


def add_past_only_history(data: pd.DataFrame) -> pd.DataFrame:
    origin = data.groupby("nameOrig", sort=False)
    destination = data.groupby("nameDest", sort=False)

    data["orig_previous_transaction_count"] = (
        origin.cumcount().astype("int32")
    )
    data["orig_previous_total_amount"] = (
        origin["amount"].cumsum() - data["amount"]
    ).astype("float32")
    origin_previous_step = origin["step"].shift(1)
    data["orig_time_since_previous"] = (
        data["step"] - origin_previous_step
    ).fillna(-1).astype("float32")

    data["dest_previous_transaction_count"] = (
        destination.cumcount().astype("int32")
    )
    data["dest_previous_total_amount"] = (
        destination["amount"].cumsum() - data["amount"]
    ).astype("float32")
    destination_previous_step = destination["step"].shift(1)
    data["dest_time_since_previous"] = (
        data["step"] - destination_previous_step
    ).fillna(-1).astype("float32")

    return data


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
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
    amount = data["amount"].to_numpy(dtype=np.float64)
    origin_balance = data["oldbalanceOrg"].to_numpy(dtype=np.float64)
    destination_balance = data[
        "oldbalanceDest"
    ].to_numpy(dtype=np.float64)
    origin_elapsed = data[
        "orig_time_since_previous"
    ].to_numpy(dtype=np.float64)
    destination_elapsed = data[
        "dest_time_since_previous"
    ].to_numpy(dtype=np.float64)
    hour_angle = 2 * np.pi * (data["step"].to_numpy() % 24) / 24

    def log_nonnegative(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(values, 0.0))

    def log_ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        ratio = numerator / (denominator + 1.0)
        return np.log1p(np.clip(ratio, 0.0, MAX_RATIO))

    features = pd.DataFrame(
        {
            "log_amount": log_nonnegative(amount),
            "log_oldbalance_org": log_nonnegative(origin_balance),
            "log_oldbalance_dest": log_nonnegative(destination_balance),
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "log_orig_previous_transaction_count": log_nonnegative(
                origin_count
            ),
            "log_orig_previous_avg_amount": log_nonnegative(
                origin_average
            ),
            "log_orig_time_since_previous": log_nonnegative(
                np.maximum(origin_elapsed, 0.0)
            ),
            "orig_history_missing": (origin_elapsed < 0).astype(float),
            "log_dest_previous_transaction_count": log_nonnegative(
                destination_count
            ),
            "log_dest_previous_avg_amount": log_nonnegative(
                destination_average
            ),
            "log_dest_time_since_previous": log_nonnegative(
                np.maximum(destination_elapsed, 0.0)
            ),
            "dest_history_missing": (
                destination_elapsed < 0
            ).astype(float),
            "log_amount_to_origin_balance": log_ratio(
                amount,
                origin_balance,
            ),
            "log_amount_to_orig_avg": log_ratio(
                amount,
                origin_average,
            ),
            "log_amount_to_dest_avg": log_ratio(
                amount,
                destination_average,
            ),
            "type_CASH_IN": (data["type"] == "CASH_IN").astype(float),
            "type_CASH_OUT": (data["type"] == "CASH_OUT").astype(float),
            "type_DEBIT": (data["type"] == "DEBIT").astype(float),
            "type_PAYMENT": (data["type"] == "PAYMENT").astype(float),
            "type_TRANSFER": (data["type"] == "TRANSFER").astype(float),
        },
        index=data.index,
    )
    features = features.loc[:, AnomalyFeatureBuilder.FEATURE_NAMES]
    return features.astype("float32")


def assert_feature_parity(
    data: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    builder = AnomalyFeatureBuilder()
    sample = data.sample(n=min(20, len(data)), random_state=RANDOM_STATE)

    for index, row in sample.iterrows():
        transaction = TransactionInput(
            transaction_id=f"paysim_{int(row['original_index']):08d}",
            sender_id=str(row["nameOrig"]),
            receiver_id=str(row["nameDest"]),
            timestamp=PAYSIM_EPOCH
            + timedelta(hours=int(row["step"])),
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
        runtime_values = np.asarray(
            [
                builder.build(transaction)[name]
                for name in builder.FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        batch_values = features.loc[index].to_numpy(dtype=np.float32)
        if not np.allclose(runtime_values, batch_values, rtol=1e-5):
            raise AssertionError(
                f"Training/runtime feature mismatch at row {index}."
            )


def normalized_scores(
    raw_scores: np.ndarray,
    calibration_scores: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(
        calibration_scores,
        raw_scores,
        side="left",
    )
    return positions / len(calibration_scores)


def evaluate(
    split_name: str,
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> None:
    predictions = anomaly_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    false_positive_rate = fp / (fp + tn)

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(
        f"PR-AUC: {average_precision_score(labels, anomaly_scores):.4f}"
    )
    print(
        f"Precision: {precision_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall: {recall_score(labels, predictions, zero_division=0):.4f}"
    )
    print(f"F1: {f1_score(labels, predictions, zero_division=0):.4f}")
    print(f"Legitimate false-positive rate: {false_positive_rate:.4%}")
    print(f"TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)

    print(f"Loading PaySim from {data_path}")
    data = add_past_only_history(load_data(data_path))

    train_mask = (data["step"] <= 520) & (data["isFraud"] == 0)
    validation_mask = data["step"].between(521, 631)
    test_mask = data["step"] > 631

    normal_train = data.loc[train_mask]
    sample_size = min(args.max_train_samples, len(normal_train))
    normal_train = normal_train.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )
    validation = data.loc[validation_mask]
    test = data.loc[test_mask]

    print(f"Building {sample_size:,} training feature rows")
    training_features = build_feature_frame(normal_train)
    assert_feature_parity(normal_train, training_features)

    model = IsolationForest(
        n_estimators=300,
        max_samples=min(8_192, sample_size),
        contamination="auto",
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Fitting Isolation Forest")
    model.fit(training_features.to_numpy(dtype=np.float32))

    print(f"Scoring {len(validation):,} validation rows")
    validation_features = build_feature_frame(validation)
    assert_feature_parity(validation, validation_features)
    validation_raw_scores = -model.score_samples(
        validation_features.to_numpy(dtype=np.float32)
    )
    validation_labels = validation["isFraud"].to_numpy()
    normal_validation_scores = np.sort(
        validation_raw_scores[validation_labels == 0].astype("float32")
    )

    boundary = np.quantile(
        normal_validation_scores,
        1 - MAX_FALSE_POSITIVE_RATE,
        method="higher",
    )
    raw_threshold = float(np.nextafter(boundary, np.inf))
    normalized_threshold = float(
        normalized_scores(
            np.asarray([raw_threshold]),
            normal_validation_scores,
        )[0]
    )
    validation_scores = normalized_scores(
        validation_raw_scores,
        normal_validation_scores,
    )
    evaluate(
        "VALIDATION",
        validation_labels,
        validation_scores,
        normalized_threshold,
    )

    print(f"\nScoring {len(test):,} untouched test rows")
    test_features = build_feature_frame(test)
    assert_feature_parity(test, test_features)
    test_raw_scores = -model.score_samples(
        test_features.to_numpy(dtype=np.float32)
    )
    test_scores = normalized_scores(
        test_raw_scores,
        normal_validation_scores,
    )
    evaluate(
        "TEST",
        test["isFraud"].to_numpy(),
        test_scores,
        normalized_threshold,
    )

    artifact = {
        "model": model,
        "feature_names": AnomalyFeatureBuilder.FEATURE_NAMES,
        "calibration_scores": normal_validation_scores,
        "raw_threshold": raw_threshold,
        "threshold": normalized_threshold,
        "model_version": MODEL_VERSION,
        "training_metadata": {
            "dataset": DATASET,
            "random_state": RANDOM_STATE,
            "normal_training_samples": sample_size,
            "train_max_step": 520,
            "validation_steps": [521, 631],
            "test_min_step": 632,
            "max_validation_false_positive_rate": (
                MAX_FALSE_POSITIVE_RATE
            ),
        },
    }
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact_path, compress=3)
    print(f"\nSaved anomaly artifact to {args.artifact_path}")


if __name__ == "__main__":
    main()
