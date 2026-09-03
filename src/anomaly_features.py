import math

from .schemas import TransactionInput

MAX_RATIO = 1_000_000.0


class AnomalyFeatureBuilder:
    """Build stationary, behavior-oriented Isolation Forest features."""

    FEATURE_NAMES = [  # noqa: RUF012
        "log_amount",
        "log_oldbalance_org",
        "log_oldbalance_dest",
        "hour_sin",
        "hour_cos",
        "log_orig_previous_transaction_count",
        "log_orig_previous_avg_amount",
        "log_orig_time_since_previous",
        "orig_history_missing",
        "log_dest_previous_transaction_count",
        "log_dest_previous_avg_amount",
        "log_dest_time_since_previous",
        "dest_history_missing",
        "log_amount_to_origin_balance",
        "log_amount_to_orig_avg",
        "log_amount_to_dest_avg",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]

    def build(self, transaction: TransactionInput) -> dict[str, float]:
        orig_avg = self._average(
            transaction.orig_previous_total_amount,
            transaction.orig_previous_transaction_count,
        )
        dest_avg = self._average(
            transaction.dest_previous_total_amount,
            transaction.dest_previous_transaction_count,
        )
        hour_angle = 2 * math.pi * (transaction.step % 24) / 24

        return {
            "log_amount": self._log_nonnegative(transaction.amount),
            "log_oldbalance_org": self._log_nonnegative(
                transaction.oldbalanceOrg
            ),
            "log_oldbalance_dest": self._log_nonnegative(
                transaction.oldbalanceDest
            ),
            "hour_sin": math.sin(hour_angle),
            "hour_cos": math.cos(hour_angle),
            "log_orig_previous_transaction_count": self._log_nonnegative(
                transaction.orig_previous_transaction_count
            ),
            "log_orig_previous_avg_amount": self._log_nonnegative(orig_avg),
            "log_orig_time_since_previous": self._log_nonnegative(
                max(transaction.orig_time_since_previous, 0.0)
            ),
            "orig_history_missing": float(
                transaction.orig_time_since_previous < 0
            ),
            "log_dest_previous_transaction_count": self._log_nonnegative(
                transaction.dest_previous_transaction_count
            ),
            "log_dest_previous_avg_amount": self._log_nonnegative(dest_avg),
            "log_dest_time_since_previous": self._log_nonnegative(
                max(transaction.dest_time_since_previous, 0.0)
            ),
            "dest_history_missing": float(
                transaction.dest_time_since_previous < 0
            ),
            "log_amount_to_origin_balance": self._log_ratio(
                transaction.amount,
                transaction.oldbalanceOrg,
            ),
            "log_amount_to_orig_avg": self._log_ratio(
                transaction.amount,
                orig_avg,
            ),
            "log_amount_to_dest_avg": self._log_ratio(
                transaction.amount,
                dest_avg,
            ),
            "type_CASH_IN": float(transaction.type == "CASH_IN"),
            "type_CASH_OUT": float(transaction.type == "CASH_OUT"),
            "type_DEBIT": float(transaction.type == "DEBIT"),
            "type_PAYMENT": float(transaction.type == "PAYMENT"),
            "type_TRANSFER": float(transaction.type == "TRANSFER"),
        }

    @staticmethod
    def _average(total: float, count: int) -> float:
        return total / count if count > 0 else 0.0

    @staticmethod
    def _log_nonnegative(value: float) -> float:
        return math.log1p(max(value, 0.0))

    @staticmethod
    def _log_ratio(numerator: float, denominator: float) -> float:
        ratio = numerator / (denominator + 1.0)
        return math.log1p(min(max(ratio, 0.0), MAX_RATIO))
