"""Print the exact JSON response returned by FraudDetector.

Usage:
    uv run python run_detector_response.py
    uv run python run_detector_response.py --example 5
"""

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

from src.fraud_detector import FraudDetector
from src.schemas import TransactionInput, TransactionType

FIXTURE_PATH = Path("src/tests/fixtures/fraud_examples.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        type=int,
        help="Print one 1-based fixture example; omit to print all.",
    )
    return parser.parse_args()


def transaction_from_row(row: pd.Series) -> TransactionInput:
    return TransactionInput(
        step=int(row["step"]),
        type=cast(TransactionType, row["type"]),
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


def main() -> None:
    args = parse_args()
    examples = pd.read_csv(FIXTURE_PATH)

    if args.example is not None:
        if not 1 <= args.example <= len(examples):
            raise SystemExit(
                f"--example must be between 1 and {len(examples)}."
            )
        selected = examples.iloc[[args.example - 1]]
    else:
        selected = examples

    detector = FraudDetector()

    for index, row in selected.iterrows():
        example_number = int(index) + 1
        prediction = detector.predict(transaction_from_row(row))

        print(f"--- FraudDetector response: example {example_number} ---")
        print(prediction.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
