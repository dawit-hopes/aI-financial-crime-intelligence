# ml/schemas.py

from typing import Literal

from pydantic import BaseModel, Field, model_validator

TransactionType = Literal[
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


class TransactionInput(BaseModel):
    # Current transaction
    step: int = Field(..., ge=0)
    type: TransactionType
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)

    # Origin account history
    orig_previous_transaction_count: int = Field(..., ge=0)
    orig_previous_total_amount: float = Field(..., ge=0)
    orig_time_since_previous: float = Field(..., ge=-1)

    # Destination account history
    dest_previous_transaction_count: int = Field(..., ge=0)
    dest_previous_total_amount: float = Field(..., ge=0)
    dest_time_since_previous: float = Field(..., ge=-1)


class FraudReason(BaseModel):
    feature: str
    contribution: float
    direction: Literal["FRAUD", "LEGITIMATE"]
    description: str


class RuleResult(BaseModel):
    rule: str
    triggered: bool
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    description: str


class AnomalyPrediction(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    raw_anomaly_score: float
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class RiskSignalScores(BaseModel):
    model_score: float = Field(..., ge=0.0, le=100.0)
    rule_score: float = Field(..., ge=0.0, le=100.0)
    anomaly_score: float = Field(..., ge=0.0, le=100.0)
    weighted_score: float = Field(..., ge=0.0, le=100.0)
    rule_floor_applied: bool


class RiskAssessment(BaseModel):
    is_fraud: bool
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    signal_scores: RiskSignalScores
    risk_engine_version: str


class RiskConfig(BaseModel):
    model_weight: float = Field(..., ge=0.0, le=1.0)
    rule_weight: float = Field(..., ge=0.0, le=1.0)
    anomaly_weight: float = Field(..., ge=0.0, le=1.0)
    low_rule_score: float = Field(20.0, ge=0.0, le=100.0)
    medium_rule_score: float = Field(60.0, ge=0.0, le=100.0)
    high_rule_score: float = Field(100.0, ge=0.0, le=100.0)
    medium_threshold: float = Field(..., ge=0.0, le=100.0)
    high_threshold: float = Field(..., ge=0.0, le=100.0)
    low_rule_floor: float = Field(0.0, ge=0.0, le=100.0)
    medium_rule_floor: float = Field(30.0, ge=0.0, le=100.0)
    high_rule_floor: float = Field(..., ge=0.0, le=100.0)
    version: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_risk_configuration(self):
        weight_sum = (
            self.model_weight
            + self.rule_weight
            + self.anomaly_weight
        )
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("Risk signal weights must sum to 1.")
        if self.medium_threshold >= self.high_threshold:
            raise ValueError(
                "Medium threshold must be lower than high threshold."
            )
        if self.high_rule_floor < self.high_threshold:
            raise ValueError(
                "High-rule floor must reach the high-risk threshold."
            )
        if self.medium_rule_floor < self.medium_threshold:
            raise ValueError(
                "Medium-rule floor must reach the medium-risk threshold."
            )
        if not (
            self.low_rule_floor
            <= self.medium_rule_floor
            <= self.high_rule_floor
        ):
            raise ValueError(
                "Rule floors must increase with severity."
            )
        return self


class FraudPrediction(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_score: float = Field(..., ge=0.0, le=100.0)
    signal_scores: RiskSignalScores
    risk_engine_version: str
    model_version: str
    anomaly: AnomalyPrediction
    reasons: list[FraudReason] = Field(default_factory=list)
    triggered_rules: list[RuleResult] = Field(default_factory=list)


class RuleConfig(BaseModel):
    drain_relative_tolerance: float = Field(0.01, ge=0.0, le=1.0)
    high_value_transfer_amount: float = Field(1_000_000.0, ge=0.0)
