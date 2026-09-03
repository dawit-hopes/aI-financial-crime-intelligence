import json
from pathlib import Path

from .schemas import (
    AnomalyPrediction,
    NetworkPrediction,
    RiskAssessment,
    RiskConfig,
    RiskSignalScores,
    RuleResult,
)

RISK_CONFIG_PATH = (
    Path(__file__).parent / "artifacts" / "risk_config.json"
)


class RiskEngine:
    """Combine model, rule, and anomaly signals into one risk score."""

    def __init__(
        self,
        config: RiskConfig | None = None,
        config_path: Path | str = RISK_CONFIG_PATH,
    ):
        self.config_path = Path(config_path)
        self.config = config or self._load_config()

    def _load_config(self) -> RiskConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Risk configuration not found: {self.config_path}"
            )

        try:
            artifact = json.loads(
                self.config_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(
                f"Could not read risk configuration: {self.config_path}"
            ) from error

        if not isinstance(artifact, dict) or "config" not in artifact:
            raise ValueError(
                "Risk configuration artifact must contain a config object."
            )

        return RiskConfig.model_validate(artifact["config"])

    def assess(
        self,
        fraud_probability: float,
        triggered_rules: list[RuleResult],
        anomaly: AnomalyPrediction,
        network: NetworkPrediction,
    ) -> RiskAssessment:
        if not 0.0 <= fraud_probability <= 1.0:
            raise ValueError(
                "Fraud probability must be between 0 and 1."
            )

        model_score = fraud_probability * 100
        anomaly_score = anomaly.anomaly_score * 100
        network_score = network.network_score
        rule_score, rule_floor = self._rule_signal(triggered_rules)
        weighted_score = (
            model_score * self.config.model_weight
            + rule_score * self.config.rule_weight
            + anomaly_score * self.config.anomaly_weight
            + network_score * self.config.network_weight
        )
        risk_score = min(100.0, max(weighted_score, rule_floor))
        rule_floor_applied = rule_floor > weighted_score

        if risk_score >= self.config.high_threshold:
            risk_level = "HIGH"
        elif risk_score >= self.config.medium_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return RiskAssessment(
            is_fraud=risk_level == "HIGH",
            risk_score=risk_score,
            risk_level=risk_level,
            signal_scores=RiskSignalScores(
                model_score=model_score,
                rule_score=rule_score,
                anomaly_score=anomaly_score,
                network_score=network_score,
                weighted_score=weighted_score,
                rule_floor_applied=rule_floor_applied,
            ),
            risk_engine_version=self.config.version,
        )

    def _rule_signal(
        self,
        triggered_rules: list[RuleResult],
    ) -> tuple[float, float]:
        severities = {
            result.severity
            for result in triggered_rules
            if result.triggered
        }
        if "HIGH" in severities:
            return (
                self.config.high_rule_score,
                self.config.high_rule_floor,
            )
        if "MEDIUM" in severities:
            return (
                self.config.medium_rule_score,
                self.config.medium_rule_floor,
            )
        if "LOW" in severities:
            return (
                self.config.low_rule_score,
                self.config.low_rule_floor,
            )
        return 0.0, 0.0
