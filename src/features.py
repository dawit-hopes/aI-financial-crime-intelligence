# ml/features.py

from .schemas import TransactionInput


class FraudFeatureBuilder:
    FEATURE_NAMES = [  # noqa: RUF012
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "hour_of_day",
        "orig_previous_transaction_count",
        "orig_previous_total_amount",
        "orig_previous_avg_amount",
        "orig_time_since_previous",
        "dest_previous_transaction_count",
        "dest_previous_total_amount",
        "dest_previous_avg_amount",
        "dest_time_since_previous",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]

    def build(self, transaction: TransactionInput) -> dict[str, float]:

        orig_avg = (
            transaction.orig_previous_total_amount
            / transaction.orig_previous_transaction_count
            if transaction.orig_previous_transaction_count > 0
            else 0.0
        )

        dest_avg = (
            transaction.dest_previous_total_amount
            / transaction.dest_previous_transaction_count
            if transaction.dest_previous_transaction_count > 0
            else 0.0
        )

        features = {
            "step": transaction.step,
            "amount": transaction.amount,
            "oldbalanceOrg": transaction.oldbalanceOrg,
            "oldbalanceDest": transaction.oldbalanceDest,
            # Derived from step
            "hour_of_day": transaction.step % 24,
            # Historical source-account features
            "orig_previous_transaction_count": transaction.orig_previous_transaction_count,
            "orig_previous_total_amount": transaction.orig_previous_total_amount,
            "orig_previous_avg_amount": orig_avg,
            "orig_time_since_previous": transaction.orig_time_since_previous,
            # Historical destination-account features
            "dest_previous_transaction_count": transaction.dest_previous_transaction_count,
            "dest_previous_total_amount": transaction.dest_previous_total_amount,
            "dest_previous_avg_amount": dest_avg,
            "dest_time_since_previous": transaction.dest_time_since_previous,
            # Transaction type one-hot encoding
            "type_CASH_IN": int(transaction.type == "CASH_IN"),
            "type_CASH_OUT": int(transaction.type == "CASH_OUT"),
            "type_DEBIT": int(transaction.type == "DEBIT"),
            "type_PAYMENT": int(transaction.type == "PAYMENT"),
            "type_TRANSFER": int(transaction.type == "TRANSFER"),
        }

        return features
