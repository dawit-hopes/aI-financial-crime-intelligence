import pandas as pd
import pytest

from src.paysim_data import (
    NETWORK_BACKFILL_MAX_STEP,
    NETWORK_BACKFILL_MIN_STEP,
    TEST_MIN_STEP,
    TRAIN_MAX_STEP,
    VALIDATION_MAX_STEP,
    VALIDATION_MIN_STEP,
    add_past_only_history,
    build_fraud_feature_frame,
    load_data,
    split_masks,
    transaction_from_row,
)
from src.features import FraudFeatureBuilder


def test_split_masks_use_expected_step_boundaries():
    data = pd.DataFrame(
        {
            "step": [520, 521, 631, 632],
            "type": ["PAYMENT"] * 4,
            "amount": [1.0] * 4,
            "nameOrig": ["A", "B", "C", "D"],
            "nameDest": ["E", "F", "G", "H"],
            "oldbalanceOrg": [0.0] * 4,
            "oldbalanceDest": [0.0] * 4,
            "isFraud": [0, 0, 0, 1],
        }
    )
    data["original_index"] = range(len(data))

    masks = split_masks(data)

    assert masks["train"].tolist() == [True, False, False, False]
    assert masks["validation"].tolist() == [False, True, True, False]
    assert masks["test"].tolist() == [False, False, False, True]
    assert masks["network_backfill"].tolist() == [True, False, False, False]


def test_transaction_from_row_requires_history_fields_after_enrichment():
    row = pd.Series(
        {
            "original_index": 12,
            "step": 100,
            "type": "TRANSFER",
            "amount": 50.0,
            "nameOrig": "sender",
            "nameDest": "receiver",
            "oldbalanceOrg": 100.0,
            "oldbalanceDest": 0.0,
            "orig_previous_transaction_count": 2,
            "orig_previous_total_amount": 25.0,
            "orig_time_since_previous": 3.0,
            "dest_previous_transaction_count": 0,
            "dest_previous_total_amount": 0.0,
            "dest_time_since_previous": -1.0,
        }
    )

    transaction = transaction_from_row(row)

    assert transaction.transaction_id == "paysim_00000012"
    assert transaction.sender_id == "sender"
    assert transaction.receiver_id == "receiver"
    assert transaction.step == 100


def test_build_fraud_feature_frame_matches_runtime_builder():
    row = pd.Series(
        {
            "original_index": 0,
            "step": 10,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "sender",
            "nameDest": "receiver",
            "oldbalanceOrg": 500.0,
            "oldbalanceDest": 20.0,
            "orig_previous_transaction_count": 2,
            "orig_previous_total_amount": 150.0,
            "orig_time_since_previous": 4.0,
            "dest_previous_transaction_count": 1,
            "dest_previous_total_amount": 20.0,
            "dest_time_since_previous": 8.0,
        }
    )
    frame = build_fraud_feature_frame(pd.DataFrame([row]))
    runtime = FraudFeatureBuilder().build(transaction_from_row(row))

    for name in FraudFeatureBuilder.FEATURE_NAMES:
        assert frame.iloc[0][name] == pytest.approx(runtime[name])


def test_temporal_constants_are_consistent():
    assert NETWORK_BACKFILL_MIN_STEP <= NETWORK_BACKFILL_MAX_STEP <= TRAIN_MAX_STEP
    assert VALIDATION_MIN_STEP == TRAIN_MAX_STEP + 1
    assert TEST_MIN_STEP == VALIDATION_MAX_STEP + 1
