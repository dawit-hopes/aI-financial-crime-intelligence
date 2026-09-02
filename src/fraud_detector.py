import shap

from .explainability import Explainability
from .features import FraudFeatureBuilder
from .model import FraudModel
from .schemas import FraudPrediction, FraudReason, TransactionInput

DEFAULT_THRESHOLD = 0.92


class FraudDetector:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.feature_builder = FraudFeatureBuilder()
        self.model = FraudModel()
        self.threshold = threshold

        # Create once when the service starts.
        self.explainer = shap.TreeExplainer(self.model.model)

        self.explainability = Explainability(
            model=self.model.model,
            explainer=self.explainer,
            threshold=self.threshold,
        )

    def predict(self, transaction: TransactionInput):
        features = self.feature_builder.build(transaction)

        probability = self.model.predict_proba(features)

        is_fraud = probability >= self.threshold

        risk_level = self._get_risk_level(probability)

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
        )

    def _get_risk_level(self, probability):
        if probability >= 0.92:
            return "HIGH"

        if probability >= 0.5:
            return "MEDIUM"

        return "LOW"
