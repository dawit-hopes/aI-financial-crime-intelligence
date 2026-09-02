import pandas as pd

from src.fraud_detector import FraudDetector
from src.schemas import TransactionInput

FIXTURE_PATH = "src/tests/fixtures/fraud_examples.csv"


def load_fixture():
    return pd.read_csv(FIXTURE_PATH)


def make_transaction(row) -> TransactionInput:
    return TransactionInput(
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


def test_fixture_contains_expected_classes():
    df = load_fixture()

    assert len(df) == 20
    assert df["isFraud"].sum() == 10
    assert (df["isFraud"] == 0).sum() == 10


def test_detector_can_predict_all_examples():
    df = load_fixture()
    detector = FraudDetector()

    for _, row in df.iterrows():
        transaction = make_transaction(row)
        result = detector.predict(transaction)

        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
        assert isinstance(result.is_fraud, bool)


def test_real_fraud_examples_are_detected():
    df = load_fixture()
    fraud_df = df[df["isFraud"] == 1]

    detector = FraudDetector()

    for _, row in fraud_df.iterrows():
        transaction = make_transaction(row)
        result = detector.predict(transaction)

        assert result.is_fraud is True, (
            f"Fraud transaction at step {row['step']} "
            f"was predicted as legitimate "
            f"(probability={result.fraud_probability:.4f})"
        )


def test_real_legitimate_examples_are_not_detected_as_fraud():
    df = load_fixture()
    legitimate_df = df[df["isFraud"] == 0]

    detector = FraudDetector()

    for _, row in legitimate_df.iterrows():
        transaction = make_transaction(row)
        result = detector.predict(transaction)

        assert result.is_fraud is False, (
            f"Legitimate transaction at step {row['step']} "
            f"was predicted as fraud "
            f"(probability={result.fraud_probability:.4f})"
        )
