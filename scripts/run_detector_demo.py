"""Run the fraud detector against the included sample transactions.

Usage:
    uv run python -m scripts.run_detector_demo
"""

from pathlib import Path
from typing import cast

import pandas as pd

from src.fraud_detector import FraudDetector
from src.schemas import TransactionInput, TransactionType

FIXTURE_PATH = Path("src/tests/fixtures/fraud_examples.csv")


def transaction_from_row(row: pd.Series) -> TransactionInput:
    """Convert one CSV row to the validated detector input schema."""
    return TransactionInput(
        transaction_id=str(row["transaction_id"]),
        sender_id=str(row["sender_id"]),
        receiver_id=str(row["receiver_id"]),
        timestamp=str(row["timestamp"]),
        step=int(row["step"]),
        # CSV values are untyped strings; Pydantic still validates this input
        # at runtime, while the cast tells the type checker the expected shape.
        type=cast(TransactionType, row["type"]),
        amount=float(row["amount"]),
        oldbalanceOrg=float(row["oldbalanceOrg"]),
        oldbalanceDest=float(row["oldbalanceDest"]),
        orig_previous_transaction_count=int(row["orig_previous_transaction_count"]),
        orig_previous_total_amount=float(row["orig_previous_total_amount"]),
        orig_time_since_previous=float(row["orig_time_since_previous"]),
        dest_previous_transaction_count=int(row["dest_previous_transaction_count"]),
        dest_previous_total_amount=float(row["dest_previous_total_amount"]),
        dest_time_since_previous=float(row["dest_time_since_previous"]),
    )


def main() -> None:
    examples = pd.read_csv(FIXTURE_PATH).sort_values(
        ["timestamp", "transaction_id"]
    )
    detector = FraudDetector()
    correct_predictions = 0

    print(f"Testing {len(examples)} sample transactions (threshold: {detector.threshold:.0%})\n")

    for number, (_, row) in enumerate(examples.iterrows(), start=1):
        prediction = detector.predict(transaction_from_row(row))
        expected = bool(row["isFraud"])
        is_correct = prediction.is_fraud == expected
        correct_predictions += is_correct

        print(f"Example {number}: {row['type']} of {row['amount']:,.2f}")
        print(
            f"  Expected: {'FRAUD' if expected else 'LEGITIMATE'} | "
            f"Predicted: {'FRAUD' if prediction.is_fraud else 'LEGITIMATE'} "
            f"(risk {prediction.risk_score:.1f}/100, "
            f"{prediction.risk_level}) | "
            f"{'CORRECT' if is_correct else 'INCORRECT'}"
        )
        print(
            f"  Signals: model={prediction.signal_scores.model_score:.1f}, "
            f"rules={prediction.signal_scores.rule_score:.1f}, "
            f"anomaly={prediction.signal_scores.anomaly_score:.1f}, "
            f"network={prediction.signal_scores.network_score:.1f}"
        )
        print(
            f"  Anomaly: {prediction.anomaly.anomaly_score:.2%} | "
            f"{'FLAGGED' if prediction.anomaly.is_anomaly else 'NOT FLAGGED'}"
        )
        if prediction.triggered_rules:
            rule_names = ", ".join(
                rule.rule for rule in prediction.triggered_rules
            )
            print(f"  Triggered rules: {rule_names}")
        if prediction.network.evidence:
            network_codes = ", ".join(
                item.code for item in prediction.network.evidence
            )
            print(f"  Network evidence: {network_codes}")
        print("  Decision reasons:")
        for reason in prediction.decision_reasons:
            print(f"    - [{reason.source}] {reason.description}")
        print("  Main reasons:")
        for reason in prediction.reasons:
            print(f"    - [{reason.direction}] {reason.description}")
        print()

    accuracy = correct_predictions / len(examples)
    print(f"Summary: {correct_predictions}/{len(examples)} correct ({accuracy:.1%} accuracy)")


if __name__ == "__main__":
    main()
