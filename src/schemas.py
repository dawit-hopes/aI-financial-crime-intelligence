# ml/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    # Current transaction
    step: int = Field(..., ge=0)
    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]
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
    impact: float
    description: str


class FraudPrediction(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    reasons: list[FraudReason] = Field(default_factory=list)
