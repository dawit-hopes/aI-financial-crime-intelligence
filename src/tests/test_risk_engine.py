import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from calibrate_risk_engine import (
    assert_batch_parity,
    build_fraud_feature_frame,
    build_high_rule_mask,
    combined_scores,
    score_signals,
)
from src.anomaly_detector import AnomalyDetector
from src.anomaly_features import AnomalyFeatureBuilder
from src.features import FraudFeatureBuilder
from src.model import FraudModel
from src.network_analyzer import NetworkAnalyzer
from src.risk_engine import RiskEngine
from src.rules import FraudRulesEngine
from src.schemas import (
    AnomalyPrediction,
    NetworkPrediction,
    NetworkConfig,
    RiskConfig,
    RuleResult,
    TransactionInput,
)
from train_anomaly_model import build_feature_frame


def make_config(**overrides) -> RiskConfig:
    values = {
        "model_weight": 0.60,
        "rule_weight": 0.20,
        "anomaly_weight": 0.15,
        "network_weight": 0.05,
        "medium_threshold": 15.0,
        "high_threshold": 55.0,
        "medium_rule_floor": 30.0,
        "high_rule_floor": 70.0,
        "version": "test",
    }
    values.update(overrides)
    return RiskConfig(**values)


def make_anomaly(score: float = 0.0) -> AnomalyPrediction:
    return AnomalyPrediction(
        is_anomaly=score >= 0.99,
        anomaly_score=score,
        raw_anomaly_score=score,
        threshold=0.99,
        model_version="test",
    )


def make_network(score: float = 0.0) -> NetworkPrediction:
    return NetworkPrediction(
        network_score=score,
        model_version="test",
    )


def make_rule(
    severity: str = "HIGH",
    triggered: bool = True,
) -> RuleResult:
    return RuleResult(
        rule="TEST_RULE",
        triggered=triggered,
        severity=severity,
        description="Test rule.",
    )


def test_risk_config_requires_weights_to_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1"):
        make_config(model_weight=0.5)


def test_risk_config_requires_ordered_thresholds():
    with pytest.raises(ValidationError, match="lower"):
        make_config(medium_threshold=55.0)


def test_risk_config_requires_rule_floors_to_reach_thresholds():
    with pytest.raises(ValidationError, match="High-rule floor"):
        make_config(high_rule_floor=54.0)

    with pytest.raises(ValidationError, match="Medium-rule floor"):
        make_config(medium_rule_floor=14.0)


def test_risk_engine_requires_existing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        RiskEngine(config_path=tmp_path / "missing.json")


def test_risk_engine_rejects_malformed_config_artifact(tmp_path):
    config_path = tmp_path / "risk.json"
    config_path.write_text(json.dumps({"wrong": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="config object"):
        RiskEngine(config_path=config_path)


def test_no_signals_produce_low_zero_risk():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.0,
        triggered_rules=[],
        anomaly=make_anomaly(0.0),
        network=make_network(),
    )

    assert result.risk_score == 0.0
    assert result.risk_level == "LOW"
    assert result.is_fraud is False
    assert result.signal_scores.model_score == 0.0
    assert result.signal_scores.rule_score == 0.0
    assert result.signal_scores.anomaly_score == 0.0
    assert result.signal_scores.rule_floor_applied is False


def test_weighted_score_combines_all_signal_values():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.5,
        triggered_rules=[],
        anomaly=make_anomaly(0.5),
        network=make_network(),
    )

    assert result.signal_scores.model_score == 50.0
    assert result.signal_scores.anomaly_score == 50.0
    assert result.signal_scores.weighted_score == pytest.approx(37.5)
    assert result.risk_score == pytest.approx(37.5)
    assert result.risk_level == "MEDIUM"


def test_model_threshold_can_drive_high_risk_without_other_signals():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.92,
        triggered_rules=[],
        anomaly=make_anomaly(0.0),
        network=make_network(),
    )

    assert result.risk_score == pytest.approx(55.2)
    assert result.risk_level == "HIGH"
    assert result.is_fraud is True


def test_anomaly_alone_can_raise_medium_but_not_high():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.0,
        triggered_rules=[],
        anomaly=make_anomaly(1.0),
        network=make_network(),
    )

    assert result.risk_score == pytest.approx(15.0)
    assert result.risk_level == "MEDIUM"
    assert result.is_fraud is False


def test_network_signal_contributes_without_declaring_fraud():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.0,
        triggered_rules=[],
        anomaly=make_anomaly(0.0),
        network=make_network(100.0),
    )

    assert result.signal_scores.network_score == 100.0
    assert result.signal_scores.weighted_score == pytest.approx(5.0)
    assert result.risk_level == "LOW"
    assert result.is_fraud is False


def test_high_rule_applies_floor_and_overrides_other_signals():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.0,
        triggered_rules=[make_rule("HIGH")],
        anomaly=make_anomaly(0.0),
        network=make_network(),
    )

    assert result.signal_scores.rule_score == 100.0
    assert result.signal_scores.weighted_score == pytest.approx(20.0)
    assert result.signal_scores.rule_floor_applied is True
    assert result.risk_score == 70.0
    assert result.risk_level == "HIGH"
    assert result.is_fraud is True


def test_medium_rule_applies_medium_floor():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.0,
        triggered_rules=[make_rule("MEDIUM")],
        anomaly=make_anomaly(0.0),
        network=make_network(),
    )

    assert result.signal_scores.rule_score == 60.0
    assert result.signal_scores.weighted_score == pytest.approx(12.0)
    assert result.signal_scores.rule_floor_applied is True
    assert result.risk_score == 30.0
    assert result.risk_level == "MEDIUM"


def test_non_triggered_rules_are_ignored():
    result = RiskEngine(make_config()).assess(
        fraud_probability=0.0,
        triggered_rules=[make_rule("HIGH", triggered=False)],
        anomaly=make_anomaly(0.0),
        network=make_network(),
    )

    assert result.risk_score == 0.0
    assert result.risk_level == "LOW"


def test_risk_score_is_clipped_to_one_hundred():
    config = make_config(
        model_weight=1.0,
        rule_weight=0.0,
        anomaly_weight=0.0,
        network_weight=0.0,
    )
    result = RiskEngine(config).assess(
        fraud_probability=1.0,
        triggered_rules=[make_rule("HIGH")],
        anomaly=make_anomaly(1.0),
        network=make_network(100.0),
    )

    assert result.risk_score == 100.0


def test_risk_engine_rejects_invalid_fraud_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        RiskEngine(make_config()).assess(
            fraud_probability=1.1,
            triggered_rules=[],
            anomaly=make_anomaly(),
            network=make_network(),
        )


def test_default_calibrated_configuration_loads():
    engine = RiskEngine()

    assert engine.config.version == "2.0.0"
    assert engine.config.high_threshold == 55.0
    assert engine.config.model_weight == 0.6


def test_calibration_batch_logic_matches_runtime_logic():
    data = pd.DataFrame(
        [
            {
                "step": 100,
                "type": "CASH_OUT",
                "amount": 100.0,
                "oldbalanceOrg": 100.0,
                "oldbalanceDest": 50.0,
                "orig_previous_transaction_count": 1,
                "orig_previous_total_amount": 75.0,
                "orig_time_since_previous": 2.0,
                "dest_previous_transaction_count": 3,
                "dest_previous_total_amount": 150.0,
                "dest_time_since_previous": 4.0,
            },
            {
                "step": 101,
                "type": "PAYMENT",
                "amount": 25.0,
                "oldbalanceOrg": 500.0,
                "oldbalanceDest": 20.0,
                "orig_previous_transaction_count": 0,
                "orig_previous_total_amount": 0.0,
                "orig_time_since_previous": -1.0,
                "dest_previous_transaction_count": 0,
                "dest_previous_total_amount": 0.0,
                "dest_time_since_previous": -1.0,
            },
        ]
    )
    fraud_features = build_fraud_feature_frame(data)
    anomaly_features = build_feature_frame(data)
    high_rule_mask = build_high_rule_mask(data)

    assert_batch_parity(
        data,
        fraud_features,
        anomaly_features,
        high_rule_mask,
    )
    assert list(anomaly_features) == AnomalyFeatureBuilder.FEATURE_NAMES
    assert high_rule_mask.tolist() == [True, False]


def test_calibration_score_formula_matches_runtime_engine():
    config = make_config()
    model_scores = np.asarray([50.0])
    anomaly_scores = np.asarray([50.0])
    high_rule_mask = np.asarray([False])
    network_scores = np.asarray([0.0])

    batch_scores, batch_weighted = combined_scores(
        model_scores,
        anomaly_scores,
        high_rule_mask,
        network_scores,
        (
            config.model_weight,
            config.rule_weight,
            config.anomaly_weight,
            config.network_weight,
        ),
    )
    runtime = RiskEngine(config).assess(
        fraud_probability=0.5,
        triggered_rules=[],
        anomaly=make_anomaly(0.5),
        network=make_network(),
    )

    assert batch_scores[0] == pytest.approx(runtime.risk_score)
    assert batch_weighted[0] == pytest.approx(
        runtime.signal_scores.weighted_score
    )


def test_calibration_signal_scores_match_runtime_components():
    data = pd.DataFrame(
        [
            {
                "step": 694,
                "type": "TRANSFER",
                "amount": 10_000_000.0,
                "oldbalanceOrg": 18_980_853.88,
                "oldbalanceDest": 0.0,
                "orig_previous_transaction_count": 0,
                "orig_previous_total_amount": 0.0,
                "orig_time_since_previous": -1.0,
                "dest_previous_transaction_count": 0,
                "dest_previous_total_amount": 0.0,
                "dest_time_since_previous": -1.0,
            }
        ]
    )
    fraud_model = FraudModel()
    anomaly_detector = AnomalyDetector()
    (
        model_scores,
        anomaly_scores,
        high_rule_mask,
        network_scores,
    ) = score_signals(
        data,
        fraud_model,
        anomaly_detector,
        NetworkAnalyzer(config=NetworkConfig()),
    )
    transaction = transaction_from_frame(data)
    runtime_features = FraudFeatureBuilder().build(transaction)
    runtime_probability = fraud_model.predict_proba(runtime_features)
    runtime_anomaly = anomaly_detector.predict(transaction)
    runtime_high_rule = any(
        result.severity == "HIGH"
        for result in FraudRulesEngine().evaluate(transaction)
    )

    assert model_scores[0] == pytest.approx(
        runtime_probability * 100,
        rel=1e-5,
    )
    assert anomaly_scores[0] == pytest.approx(
        runtime_anomaly.anomaly_score * 100,
        rel=1e-5,
    )
    assert bool(high_rule_mask[0]) is runtime_high_rule
    assert 0.0 <= network_scores[0] <= 100.0


def transaction_from_frame(data: pd.DataFrame) -> TransactionInput:
    row = data.iloc[0]

    return TransactionInput(
        transaction_id="test_transaction",
        sender_id="sender",
        receiver_id="receiver",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        step=int(row["step"]),
        type=row["type"],
        amount=float(row["amount"]),
        oldbalanceOrg=float(row["oldbalanceOrg"]),
        oldbalanceDest=float(row["oldbalanceDest"]),
        orig_previous_transaction_count=int(
            row["orig_previous_transaction_count"]
        ),
        orig_previous_total_amount=float(
            row["orig_previous_total_amount"]
        ),
        orig_time_since_previous=float(
            row["orig_time_since_previous"]
        ),
        dest_previous_transaction_count=int(
            row["dest_previous_transaction_count"]
        ),
        dest_previous_total_amount=float(
            row["dest_previous_total_amount"]
        ),
        dest_time_since_previous=float(
            row["dest_time_since_previous"]
        ),
    )
