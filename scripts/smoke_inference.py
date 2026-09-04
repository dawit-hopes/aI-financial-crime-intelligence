"""Load artifacts and score one transaction. Used as the Docker image CMD."""

from datetime import UTC, datetime

from src.fraud_detector import FraudDetector
from src.schemas import TransactionInput


def main() -> None:
    detector = FraudDetector()
    prediction = detector.predict(
        TransactionInput(
            transaction_id="smoke-001",
            sender_id="smoke-sender",
            receiver_id="smoke-receiver",
            timestamp=datetime(2026, 1, 30, 1, tzinfo=UTC),
            step=697,
            type="CASH_OUT",
            amount=378_623.90,
            oldbalanceOrg=378_623.90,
            oldbalanceDest=0.0,
            orig_previous_transaction_count=0,
            orig_previous_total_amount=0.0,
            orig_time_since_previous=-1.0,
            dest_previous_transaction_count=0,
            dest_previous_total_amount=0.0,
            dest_time_since_previous=-1.0,
        )
    )
    print(
        "smoke_ok "
        f"risk_level={prediction.risk_level} "
        f"risk_score={prediction.risk_score:.1f} "
        f"model_version={prediction.model_version} "
        f"risk_engine_version={prediction.risk_engine_version}"
    )


if __name__ == "__main__":
    main()
