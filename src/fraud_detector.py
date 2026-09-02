from typing import Literal

from .features import FraudFeatureBuilder
from .model import FraudModel
from .schemas import FraudPrediction, FraudReason, TransactionInput


class FraudDetector:
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        self.feature_builder = FraudFeatureBuilder()
        self.model = FraudModel()
        self.threshold = threshold

    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        """Run fraud detection on a single transaction."""

        # 1. Build model features
        features = self.feature_builder.build(transaction)

        # 2. Get fraud probability
        probability = self.model.predict_proba(features)

        # 3. Determine fraud classification
        is_fraud = probability >= self.threshold

        # 4. Determine risk level
        risk_level = self._get_risk_level(probability)

        # 5. Generate explanation
        reasons = self._generate_reasons(features, probability)

        return FraudPrediction(
            is_fraud=is_fraud,
            fraud_probability=probability,
            risk_level=risk_level,
            model_version="1.0.0",
            reasons=reasons,
        )

    def _get_risk_level(
        self,
        probability: float,
    ) -> Literal["LOW", "MEDIUM", "HIGH"]:
        if probability >= 0.8:
            return "HIGH"

        if probability >= 0.5:
            return "MEDIUM"

        return "LOW"

    def _generate_reasons(
        self,
        features: dict[str, float],
        probability: float,
    ) -> list[FraudReason]:
        reasons = []

        if probability < 0.5:
            return reasons

        if features["amount"] > 100000:
            reasons.append(
                FraudReason(
                    feature="amount",
                    impact=0.0,
                    description="Transaction amount is unusually large.",
                )
            )

        if features["type_TRANSFER"] == 1:
            reasons.append(
                FraudReason(
                    feature="type_TRANSFER",
                    impact=0.0,
                    description="Transaction is a transfer.",
                )
            )

        if features["orig_previous_transaction_count"] == 0:
            reasons.append(
                FraudReason(
                    feature="orig_previous_transaction_count",
                    impact=0.0,
                    description="No previous transaction history is available for the origin account.",
                )
            )

        if (
            features["orig_previous_transaction_count"] > 0
            and features["orig_previous_avg_amount"] > 0
            and features["amount"] > features["orig_previous_avg_amount"] * 5
        ):
            reasons.append(
                FraudReason(
                    feature="orig_previous_avg_amount",
                    impact=0.0,
                    description="Transaction amount is significantly higher than the origin account's historical average.",
                )
            )

        if (
            features["dest_previous_transaction_count"] == 0
            and features["type_TRANSFER"] == 1
        ):
            reasons.append(
                FraudReason(
                    feature="dest_previous_transaction_count",
                    impact=0.0,
                    description="Transfer is being sent to a destination account with no previous transaction history.",
                )
            )

        return reasons
