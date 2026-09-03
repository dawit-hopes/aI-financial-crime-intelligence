from .schemas import RuleConfig, RuleResult, TransactionInput


class FraudRulesEngine:
    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()

    def evaluate(self, transaction: TransactionInput) -> list[RuleResult]:
        """Return the rules triggered by a transaction."""
        return [
            result
            for result in self.evaluate_all(transaction)
            if result.triggered
        ]

    def evaluate_all(self, transaction: TransactionInput) -> list[RuleResult]:
        """Return every rule outcome for diagnostics and testing."""
        return [
            self._origin_account_drain(transaction),
            self._high_value_funded_transfer(transaction),
        ]

    def _origin_account_drain(
        self,
        transaction: TransactionInput,
    ) -> RuleResult:
        fraud_prone_type = transaction.type in {"CASH_OUT", "TRANSFER"}
        has_origin_balance = transaction.oldbalanceOrg > 0
        allowed_difference = (
            transaction.oldbalanceOrg
            * self.config.drain_relative_tolerance
        )
        balance_difference = abs(
            transaction.amount - transaction.oldbalanceOrg
        )

        triggered = (
            fraud_prone_type
            and has_origin_balance
            and balance_difference <= allowed_difference
        )

        return RuleResult(
            rule="ORIGIN_ACCOUNT_DRAIN",
            triggered=triggered,
            severity="HIGH" if triggered else "LOW",
            description=(
                f"The {transaction.type} amount of "
                f"{transaction.amount:,.2f} is within "
                f"{self.config.drain_relative_tolerance:.1%} of the "
                f"origin balance of {transaction.oldbalanceOrg:,.2f}."
                if triggered
                else (
                    "The transaction does not match the configured "
                    "origin-account drain pattern."
                )
            ),
        )

    def _high_value_funded_transfer(
        self,
        transaction: TransactionInput,
    ) -> RuleResult:
        triggered = (
            transaction.type == "TRANSFER"
            and transaction.amount
            >= self.config.high_value_transfer_amount
            and transaction.oldbalanceOrg >= transaction.amount
        )

        return RuleResult(
            rule="HIGH_VALUE_FUNDED_TRANSFER",
            triggered=triggered,
            severity="HIGH" if triggered else "LOW",
            description=(
                f"The funded transfer amount of "
                f"{transaction.amount:,.2f} meets or exceeds the "
                f"configured high-value threshold of "
                f"{self.config.high_value_transfer_amount:,.2f}."
                if triggered
                else (
                    "The transaction does not match the configured "
                    "high-value funded-transfer pattern."
                )
            ),
        )