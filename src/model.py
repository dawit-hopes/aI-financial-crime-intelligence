from pathlib import Path

import pandas as pd
import xgboost as xgb

MODEL_PATH = Path(__file__).parent / "artifacts" / "fraud_model.json"


class FraudModel:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self) -> xgb.XGBClassifier:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Fraud model artifact not found: {MODEL_PATH}")

        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        return model

    def predict_proba(self, features: dict[str, float]) -> float:
        feature_vector = pd.DataFrame([features])

        probability = self.model.predict_proba(feature_vector)[0, 1]

        return float(probability)
