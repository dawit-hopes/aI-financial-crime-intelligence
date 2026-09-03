"""Shared PaySim loading, history features, and temporal splits."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from .features import FraudFeatureBuilder
from .schemas import TransactionInput

DATASET = "ealaxi/paysim1"
PAYSIM_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
RANDOM_STATE = 42

TRAIN_MAX_STEP = 520
VALIDATION_MIN_STEP = 521
VALIDATION_MAX_STEP = 631
TEST_MIN_STEP = 632
NETWORK_BACKFILL_MIN_STEP = 497
NETWORK_BACKFILL_MAX_STEP = 520

PRESERVED_DECISION_THRESHOLD = 0.92
THRESHOLD_SELECTION = "preserved_operating_point"
THRESHOLD_SELECTION_DATASET = "validation"


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


def split_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "train": data["step"] <= TRAIN_MAX_STEP,
        "validation": data["step"].between(
            VALIDATION_MIN_STEP,
            VALIDATION_MAX_STEP,
        ),
        "test": data["step"] > TEST_MIN_STEP - 1,
        "network_backfill": data["step"].between(
            NETWORK_BACKFILL_MIN_STEP,
            NETWORK_BACKFILL_MAX_STEP,
        ),
    }


def split_by_step(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = split_masks(data)
    return {
        name: data.loc[mask].copy()
        for name, mask in masks.items()
    }


def row_source_index(row: pd.Series) -> int:
    if "original_index" in row:
        return int(row["original_index"])
    if "source_row" in row:
        return int(row["source_row"])
    return int(row.name)


def transaction_from_row(row: pd.Series) -> TransactionInput:
    source_index = row_source_index(row)
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
        oldbalanceOrg=float(row.get("oldbalanceOrg", 0.0)),
        oldbalanceDest=float(row.get("oldbalanceDest", 0.0)),
        orig_previous_transaction_count=int(
            row.get("orig_previous_transaction_count", 0)
        ),
        orig_previous_total_amount=float(
            row.get("orig_previous_total_amount", 0.0)
        ),
        orig_time_since_previous=float(
            row.get("orig_time_since_previous", -1.0)
        ),
        dest_previous_transaction_count=int(
            row.get("dest_previous_transaction_count", 0)
        ),
        dest_previous_total_amount=float(
            row.get("dest_previous_total_amount", 0.0)
        ),
        dest_time_since_previous=float(
            row.get("dest_time_since_previous", -1.0)
        ),
    )


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


def assert_fraud_feature_parity(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    sample_size: int = 20,
) -> None:
    builder = FraudFeatureBuilder()
    sample = data.sample(
        n=min(sample_size, len(data)),
        random_state=RANDOM_STATE,
    )

    for index, row in sample.iterrows():
        transaction = transaction_from_row(row)
        runtime_values = np.asarray(
            [
                builder.build(transaction)[name]
                for name in FraudFeatureBuilder.FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        batch_values = features.loc[index].to_numpy(dtype=np.float32)
        if not np.allclose(runtime_values, batch_values, rtol=1e-5):
            raise AssertionError(
                f"Fraud feature parity failed at row {index}."
            )
