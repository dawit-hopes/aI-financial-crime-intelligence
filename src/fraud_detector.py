from .anomaly_detector import AnomalyDetector
from .explainability import Explainability
from .features import FraudFeatureBuilder
from .model import FraudModel
from .risk_engine import RiskEngine
from .rules import FraudRulesEngine
from .schemas import (
    FraudPrediction,
    FraudReason,
    TransactionInput,
)

DEFAULT_THRESHOLD = 0.92


class FraudDetector:

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        rules_engine: FraudRulesEngine | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        risk_engine: RiskEngine | None = None,
    ):
        self.feature_builder = FraudFeatureBuilder()
        self.model = FraudModel()
        self.threshold = threshold
        self.rules_engine = rules_engine or FraudRulesEngine()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.risk_engine = risk_engine or RiskEngine()

        self.explainability = Explainability(
            model=self.model,
            threshold=self.threshold,
        )

    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        features = self.feature_builder.build(transaction)

        probability = self.model.predict_proba(features)
        triggered_rules = self.rules_engine.evaluate(transaction)
        anomaly = self.anomaly_detector.predict(transaction)
        risk = self.risk_engine.assess(
            probability,
            triggered_rules,
            anomaly,
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
            is_fraud=risk.is_fraud,
            fraud_probability=probability,
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            signal_scores=risk.signal_scores,
            risk_engine_version=risk.risk_engine_version,
            model_version="1.0.0",
            anomaly=anomaly,
            reasons=reasons,
            triggered_rules=triggered_rules,
        )