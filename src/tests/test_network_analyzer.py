from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.network_analyzer import NetworkAnalyzer
from src.schemas import NetworkConfig, TransactionInput

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def transaction(
    transaction_id: str,
    sender: str,
    receiver: str,
    *,
    amount: float = 100.0,
    hour: int = 0,
) -> TransactionInput:
    return TransactionInput(
        transaction_id=transaction_id,
        sender_id=sender,
        receiver_id=receiver,
        timestamp=BASE_TIME + timedelta(hours=hour),
        step=hour,
        type="TRANSFER",
        amount=amount,
        oldbalanceOrg=amount,
        oldbalanceDest=0.0,
        orig_previous_transaction_count=0,
        orig_previous_total_amount=0.0,
        orig_time_since_previous=-1.0,
        dest_previous_transaction_count=0,
        dest_previous_total_amount=0.0,
        dest_time_since_previous=-1.0,
    )


def evidence_codes(result) -> set[str]:
    return {item.code for item in result.evidence}


def test_empty_graph_scores_before_recording_transaction():
    analyzer = NetworkAnalyzer(config=NetworkConfig())

    result = analyzer.analyze_and_record(transaction("tx1", "A", "B"))

    assert result.network_score == 0.0
    assert result.indicators["sender_fan_out"] == 1.0
    assert analyzer.graph.number_of_edges() == 1


def test_duplicate_transaction_is_idempotent():
    analyzer = NetworkAnalyzer(config=NetworkConfig())
    item = transaction("tx1", "A", "B")

    first = analyzer.analyze_and_record(item)
    second = analyzer.analyze_and_record(item)

    assert first == second
    assert analyzer.graph.number_of_edges() == 1


def test_sender_fan_out_and_receiver_fan_in():
    config = NetworkConfig(fan_out_threshold=2, fan_in_threshold=2)
    analyzer = NetworkAnalyzer(config=config)
    analyzer.analyze_and_record(transaction("tx1", "A", "B"))

    fan_out = analyzer.analyze_and_record(
        transaction("tx2", "A", "C", hour=1)
    )
    fan_in = analyzer.analyze_and_record(
        transaction("tx3", "D", "C", hour=2)
    )

    assert "HIGH_SENDER_FAN_OUT" in evidence_codes(fan_out)
    assert "HIGH_RECEIVER_FAN_IN" in evidence_codes(fan_in)


def test_repeated_pair_activity():
    analyzer = NetworkAnalyzer(
        config=NetworkConfig(repeated_pair_threshold=2)
    )
    analyzer.analyze_and_record(transaction("tx1", "A", "B"))

    result = analyzer.analyze_and_record(
        transaction("tx2", "A", "B", hour=1)
    )

    assert "REPEATED_PAIR_ACTIVITY" in evidence_codes(result)
    assert result.indicators["repeated_pair_count"] == 2.0


def test_rapid_pass_through():
    analyzer = NetworkAnalyzer(
        config=NetworkConfig(pass_through_ratio=0.8)
    )
    analyzer.analyze_and_record(
        transaction("tx1", "SOURCE", "A", amount=100.0)
    )

    result = analyzer.analyze_and_record(
        transaction("tx2", "A", "B", amount=80.0, hour=1)
    )

    assert "RAPID_PASS_THROUGH" in evidence_codes(result)


def test_short_cycle_closure():
    analyzer = NetworkAnalyzer(config=NetworkConfig())
    analyzer.analyze_and_record(transaction("tx1", "A", "B"))
    analyzer.analyze_and_record(transaction("tx2", "B", "C", hour=1))

    result = analyzer.analyze_and_record(
        transaction("tx3", "C", "A", hour=2)
    )

    assert "SHORT_CYCLE_CLOSURE" in evidence_codes(result)


def test_component_size_and_volume_evidence():
    analyzer = NetworkAnalyzer(
        config=NetworkConfig(
            component_size_threshold=3,
            sender_volume_threshold=150.0,
        )
    )
    analyzer.analyze_and_record(
        transaction("tx1", "A", "B", amount=100.0)
    )

    result = analyzer.analyze_and_record(
        transaction("tx2", "A", "C", amount=100.0, hour=1)
    )

    assert "LARGE_CONNECTED_COMPONENT" in evidence_codes(result)
    assert "CONCENTRATED_SENDER_VOLUME" in evidence_codes(result)


def test_old_edges_are_pruned_by_lookback():
    analyzer = NetworkAnalyzer(
        config=NetworkConfig(lookback_hours=1.0)
    )
    analyzer.analyze_and_record(transaction("tx1", "A", "B"))

    analyzer.analyze_and_record(
        transaction("tx2", "C", "D", hour=2)
    )

    assert "A" not in analyzer.graph
    assert analyzer.graph.number_of_edges() == 1


def test_edges_are_pruned_to_maximum_count():
    analyzer = NetworkAnalyzer(config=NetworkConfig(max_edges=2))
    analyzer.analyze_and_record(transaction("tx1", "A", "B"))
    analyzer.analyze_and_record(transaction("tx2", "B", "C", hour=1))
    analyzer.analyze_and_record(transaction("tx3", "C", "D", hour=2))

    assert analyzer.graph.number_of_edges() == 2


def test_backfill_sorts_transactions_and_is_idempotent():
    analyzer = NetworkAnalyzer(config=NetworkConfig())
    items = [
        transaction("tx2", "B", "C", hour=2),
        transaction("tx1", "A", "B", hour=1),
    ]

    analyzer.backfill(items)
    analyzer.backfill(items)

    assert analyzer.graph.number_of_edges() == 2
    assert analyzer._latest_timestamp == BASE_TIME + timedelta(hours=2)


def test_live_transactions_must_be_chronological():
    analyzer = NetworkAnalyzer(config=NetworkConfig())
    analyzer.analyze_and_record(transaction("tx2", "A", "B", hour=2))

    with pytest.raises(ValueError, match="chronologically"):
        analyzer.analyze_and_record(
            transaction("tx1", "B", "C", hour=1)
        )


def test_network_score_is_capped_at_one_hundred():
    config = NetworkConfig(
        fan_out_threshold=1,
        fan_in_threshold=1,
        repeated_pair_threshold=1,
        component_size_threshold=2,
        sender_volume_threshold=1.0,
        fan_out_points=100.0,
        fan_in_points=100.0,
        repeated_pair_points=100.0,
        component_points=100.0,
        volume_points=100.0,
    )

    result = NetworkAnalyzer(config=config).analyze_and_record(
        transaction("tx1", "A", "B")
    )

    assert result.network_score == 100.0


def test_transaction_contract_requires_timezone_aware_timestamp():
    with pytest.raises(ValidationError, match="timezone"):
        TransactionInput(
            **transaction("tx1", "A", "B").model_dump(
                exclude={"timestamp"}
            ),
            timestamp=datetime(2026, 1, 1),
        )
