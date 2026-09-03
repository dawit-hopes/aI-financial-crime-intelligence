from .anomaly_detector import AnomalyDetector
from .explainability import Explainability
from .features import FraudFeatureBuilder
from .model import FraudModel
from .network_analyzer import NetworkAnalyzer
from .risk_engine import RiskEngine
from .rules import FraudRulesEngine
from .schemas import (
    AnomalyPrediction,
    DecisionReason,
    FraudPrediction,
    FraudReason,
    NetworkPrediction,
    RuleResult,
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
        network_analyzer: NetworkAnalyzer | None = None,
    ):
        self.feature_builder = FraudFeatureBuilder()
        self.model = FraudModel()
        self.threshold = threshold
        self.rules_engine = rules_engine or FraudRulesEngine()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.risk_engine = risk_engine or RiskEngine()
        self.network_analyzer = network_analyzer or NetworkAnalyzer()

        self.explainability = Explainability(
            model=self.model,
            threshold=self.threshold,
        )

    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        features = self.feature_builder.build(transaction)

        probability = self.model.predict_proba(features)
        triggered_rules = self.rules_engine.evaluate(transaction)
        anomaly = self.anomaly_detector.predict(transaction)
        network = self.network_analyzer.analyze_and_record(transaction)
        risk = self.risk_engine.assess(
            probability,
            triggered_rules,
            anomaly,
            network,
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
        decision_reasons = self._decision_reasons(
            risk.is_fraud,
            reasons,
            triggered_rules,
            anomaly,
            network,
        )

        return FraudPrediction(
            is_fraud=risk.is_fraud,
            fraud_probability=probability,
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            signal_scores=risk.signal_scores,
            risk_engine_version=risk.risk_engine_version,
            model_version="1.0.0",
            anomaly=anomaly,
            network=network,
            reasons=reasons,
            decision_reasons=decision_reasons,
            triggered_rules=triggered_rules,
        )

    @staticmethod
    def _decision_reasons(
        is_fraud: bool,
        model_reasons: list[FraudReason],
        triggered_rules: list[RuleResult],
        anomaly: AnomalyPrediction,
        network: NetworkPrediction,
    ) -> list[DecisionReason]:
        reasons = [
            DecisionReason(
                source="RULE",
                code=rule.rule,
                description=rule.description,
            )
            for rule in triggered_rules
        ]
        if anomaly.is_anomaly:
            reasons.append(
                DecisionReason(
                    source="ANOMALY",
                    code="ISOLATION_FOREST_ANOMALY",
                    description=(
                        f"Transaction is in the "
                        f"{anomaly.anomaly_score:.1%} anomaly percentile."
                    ),
                )
            )
        reasons.extend(
            DecisionReason(
                source="NETWORK",
                code=item.code,
                description=item.description,
            )
            for item in network.evidence
        )
        selected_model_reasons = [
            reason
            for reason in model_reasons
            if not is_fraud or reason.direction == "FRAUD"
        ][:3]
        reasons.extend(
            DecisionReason(
                source="MODEL",
                code=reason.feature,
                description=reason.description,
            )
            for reason in selected_model_reasons
        )
        return reasons