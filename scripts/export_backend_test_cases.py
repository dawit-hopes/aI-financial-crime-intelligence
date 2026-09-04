"""Export a held-out PaySim test-split pack for backend integration testing.

Reuses the same preparation as the notebook fixtures:
load PaySim, restrict to the test split, then sample balanced
fraud/legitimate rows. Existing unit-test fixtures are excluded so
this pack is a separate set of cases.

The exported fields are the current-transaction payload the backend
sends, plus the ground-truth `isFraud` flag. History features are not
included.

Usage:
    uv run python -m scripts.export_backend_test_cases
    uv run python -m scripts.export_backend_test_cases --data-path /path/to/paysim.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.paysim_data import (
    load_data,
    resolve_data_path,
    split_by_step,
    transaction_from_row,
)

EXISTING_FIXTURE_PATH = Path("src/tests/fixtures/fraud_examples.csv")
DEFAULT_CSV_PATH = Path("src/tests/fixtures/backend_test_cases.csv")
DEFAULT_JSON_PATH = Path("src/tests/fixtures/backend_test_cases.json")
EXPORT_COLUMNS = [
    "transaction_id",
    "sender_id",
    "receiver_id",
    "timestamp",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]
FRAUD_COUNT = 15
LEGIT_COUNT = 15
SAMPLE_RANDOM_STATE = 2026
MONEY_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args()


def format_timestamp(timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def export_payload(row: pd.Series) -> dict:
    transaction = transaction_from_row(row)
    return {
        "transaction_id": transaction.transaction_id,
        "sender_id": transaction.sender_id,
        "receiver_id": transaction.receiver_id,
        "timestamp": format_timestamp(transaction.timestamp),
        "step": transaction.step,
        "type": transaction.type,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "oldbalanceDest": transaction.oldbalanceDest,
        "isFraud": int(row["isFraud"]),
    }


def sample_class(
    data: pd.DataFrame,
    *,
    is_fraud: int,
    count: int,
    random_state: int,
    cover_types: bool = False,
) -> pd.DataFrame:
    subset = data[data["isFraud"] == is_fraud]
    if len(subset) < count:
        raise ValueError(
            f"Need {count} isFraud={is_fraud} rows, found {len(subset)}."
        )
    if not cover_types:
        return subset.sample(n=count, random_state=random_state)

    covered = []
    remaining = subset
    for transaction_type in sorted(remaining["type"].astype(str).unique()):
        type_rows = remaining[remaining["type"].astype(str) == transaction_type]
        if type_rows.empty:
            continue
        chosen = type_rows.sample(n=1, random_state=random_state)
        covered.append(chosen)
        remaining = remaining.drop(index=chosen.index)

    covered_frame = pd.concat(covered)
    missing = count - len(covered_frame)
    if missing > 0:
        extra = remaining.sample(n=missing, random_state=random_state)
        covered_frame = pd.concat([covered_frame, extra])
    return covered_frame


def round_money(data: pd.DataFrame) -> pd.DataFrame:
    rounded = data.copy()
    for column in MONEY_COLUMNS:
        rounded[column] = rounded[column].astype("float64").round(2)
    return rounded


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    existing_ids = set(
        pd.read_csv(EXISTING_FIXTURE_PATH)["transaction_id"].astype(str)
    )

    data = load_data(data_path)
    test = split_by_step(data)["test"].copy()
    test["transaction_id"] = test["original_index"].map(
        lambda index: f"paysim_{int(index):08d}"
    )
    remaining = test[~test["transaction_id"].isin(existing_ids)]

    selected = round_money(
        pd.concat(
            [
                sample_class(
                    remaining,
                    is_fraud=1,
                    count=FRAUD_COUNT,
                    random_state=SAMPLE_RANDOM_STATE,
                ),
                sample_class(
                    remaining,
                    is_fraud=0,
                    count=LEGIT_COUNT,
                    random_state=SAMPLE_RANDOM_STATE,
                    cover_types=True,
                ),
            ]
        ).sort_values(["step", "original_index"], kind="stable")
    )

    rows = [export_payload(row) for _, row in selected.iterrows()]
    csv_frame = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_frame.to_csv(args.csv_path, index=False)
    args.json_path.write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(csv_frame)} cases to {args.csv_path}")
    print(f"Saved JSON pack to {args.json_path}")
    print("Class distribution:")
    print(csv_frame["isFraud"].value_counts().to_string())
    print("Types:")
    print(csv_frame.groupby(["isFraud", "type"]).size().to_string())


if __name__ == "__main__":
    main()
