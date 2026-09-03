from pathlib import Path

import numpy as np
import xgboost as xgb

from .features import FraudFeatureBuilder

MODEL_PATH = Path(__file__).parent / "artifacts" / "fraud_model.json"


class FraudModel:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self) -> xgb.Booster:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Fraud model artifact not found: {MODEL_PATH}")

        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        if model.feature_names != FraudFeatureBuilder.FEATURE_NAMES:
            raise ValueError(
                "Model artifact feature order does not match " "the feature builder."
            )

        return model

    @staticmethod
    def _to_dmatrix(
        features: dict[str, float],
    ) -> xgb.DMatrix:
        names = FraudFeatureBuilder.FEATURE_NAMES

        if set(features) != set(names):
            raise ValueError("Feature set does not match the trained model.")

        values = np.asarray(
            [[features[name] for name in names]],
            dtype=np.float32,
        )

        return xgb.DMatrix(
            values,
            feature_names=names,
        )

    def predict_proba(
        self,
        features: dict[str, float],
    ) -> float:
        dmatrix = self._to_dmatrix(features)

        probability = self.model.predict(dmatrix)[0]

        return float(probability)

    def feature_contributions(
        self,
        features: dict[str, float],
    ) -> np.ndarray:
        dmatrix = self._to_dmatrix(features)

        contributions = self.model.predict(
            dmatrix,
            pred_contribs=True,
        )[0]

        return contributions[:-1]
