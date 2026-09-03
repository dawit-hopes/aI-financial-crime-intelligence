from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .anomaly_features import AnomalyFeatureBuilder
from .schemas import AnomalyPrediction, TransactionInput

ANOMALY_MODEL_PATH = (
    Path(__file__).parent / "artifacts" / "anomaly_model.joblib"
)
REQUIRED_ARTIFACT_KEYS = {
    "model",
    "feature_names",
    "calibration_scores",
    "raw_threshold",
    "threshold",
    "model_version",
}


class AnomalyDetector:
    """Score transactions against learned normal PaySim behavior."""

    def __init__(
        self,
        artifact_path: Path | str = ANOMALY_MODEL_PATH,
    ):
        self.artifact_path = Path(artifact_path)
        self.feature_builder = AnomalyFeatureBuilder()
        self._load_artifact()

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Anomaly model artifact not found: {self.artifact_path}"
            )

        artifact: Any = joblib.load(self.artifact_path)
        if not isinstance(artifact, dict):
            raise ValueError("Anomaly artifact must be a dictionary.")

        missing_keys = REQUIRED_ARTIFACT_KEYS - artifact.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"Anomaly artifact is missing required keys: {missing}"
            )

        feature_names = artifact["feature_names"]
        if feature_names != self.feature_builder.FEATURE_NAMES:
            raise ValueError(
                "Anomaly artifact feature order does not match "
                "the anomaly feature builder."
            )

        calibration_scores = np.asarray(
            artifact["calibration_scores"],
            dtype=np.float32,
        )
        if (
            calibration_scores.ndim != 1
            or calibration_scores.size == 0
            or not np.isfinite(calibration_scores).all()
            or np.any(np.diff(calibration_scores) < 0)
        ):
            raise ValueError(
                "Anomaly calibration scores must be a non-empty, "
                "finite, sorted one-dimensional array."
            )

        threshold = float(artifact["threshold"])
        raw_threshold = float(artifact["raw_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Anomaly threshold must be between 0 and 1."
            )
        if not np.isfinite(raw_threshold):
            raise ValueError("Raw anomaly threshold must be finite.")
        if not hasattr(artifact["model"], "score_samples"):
            raise ValueError(
                "Anomaly artifact model must provide score_samples()."
            )

        self.model = artifact["model"]
        self.calibration_scores = calibration_scores
        self.raw_threshold = raw_threshold
        self.threshold = threshold
        self.model_version = str(artifact["model_version"])

    def predict(
        self,
        transaction: TransactionInput,
    ) -> AnomalyPrediction:
        features = self.feature_builder.build(transaction)
        values = np.asarray(
            [
                [features[name] for name in self.feature_builder.FEATURE_NAMES]
            ],
            dtype=np.float32,
        )
        raw_anomaly_score = -float(self.model.score_samples(values)[0])
        position = np.searchsorted(
            self.calibration_scores,
            raw_anomaly_score,
            side="left",
        )
        anomaly_score = float(
            np.clip(position / len(self.calibration_scores), 0.0, 1.0)
        )

        return AnomalyPrediction(
            is_anomaly=anomaly_score >= self.threshold,
            anomaly_score=anomaly_score,
            raw_anomaly_score=raw_anomaly_score,
            threshold=self.threshold,
            model_version=self.model_version,
        )
