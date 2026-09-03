from .explainability import Explainability
from .features import FraudFeatureBuilder
from .model import FraudModel
from .rules import FraudRulesEngine
from .schemas import (
    FraudPrediction,
    FraudReason,
    RuleResult,
    TransactionInput,
)

DEFAULT_THRESHOLD = 0.92


class FraudDetector:

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        rules_engine: FraudRulesEngine | None = None,
    ):
        self.feature_builder = FraudFeatureBuilder()
        self.model = FraudModel()
        self.threshold = threshold
        self.rules_engine = rules_engine or FraudRulesEngine()

        self.explainability = Explainability(
            model=self.model,
            threshold=self.threshold,
        )

    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        features = self.feature_builder.build(transaction)

        probability = self.model.predict_proba(features)
        triggered_rules = self.rules_engine.evaluate(transaction)
        has_high_risk_rule = any(
            result.severity == "HIGH"
            for result in triggered_rules
        )

        is_fraud = (
            probability >= self.threshold
            or has_high_risk_rule
        )
        risk_level = self._get_risk_level(
            probability,
            triggered_rules,
        )

        explanation = self.explainability.explain_transaction(
            features=features,
            top_n=5,
        )

        reasons = [
            FraudReason(
                feature=reason["feature"],
                contribution=reason["contribution"],
                direction=reason["direction"],
                description=reason["description"],
            )
            for reason in explanation["reasons"]
        ]

        return FraudPrediction(
            is_fraud=is_fraud,
            fraud_probability=probability,
            risk_level=risk_level,
            model_version="1.0.0",
            reasons=reasons,
            triggered_rules=triggered_rules,
        )

    def _get_risk_level(
        self,
        probability: float,
        triggered_rules: list[RuleResult],
    ):
        severities = {
            result.severity
            for result in triggered_rules
        }

        if probability >= self.threshold or "HIGH" in severities:
            return "HIGH"

        if probability >= 0.5 or "MEDIUM" in severities:
            return "MEDIUM"

        return "LOW"