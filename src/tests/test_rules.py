import csv
from pathlib import Path

import pytest

from src.rules import FraudRulesEngine
from src.schemas import RuleConfig, TransactionInput, TransactionType

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fraud_examples.csv"


def make_transaction(
    *,
    transaction_type: TransactionType = "PAYMENT",
    amount: float = 100.0,
    origin_balance: float = 1_000.0,
) -> TransactionInput:
    return TransactionInput(
        step=1,
        type=transaction_type,
        amount=amount,
        oldbalanceOrg=origin_balance,
        oldbalanceDest=0.0,
        orig_previous_transaction_count=0,
        orig_previous_total_amount=0.0,
        orig_time_since_previous=-1.0,
        dest_previous_transaction_count=0,
        dest_previous_total_amount=0.0,
        dest_time_since_previous=-1.0,
    )


def load_fixture() -> list[tuple[TransactionInput, bool]]:
    transactions = []

    with FIXTURE_PATH.open(newline="") as fixture:
        for row in csv.DictReader(fixture):
            transactions.append(
                (
                    TransactionInput(
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
                    ),
                    bool(int(row["isFraud"])),
                )
            )

    return transactions


@pytest.mark.parametrize("transaction_type", ["CASH_OUT", "TRANSFER"])
def test_origin_account_drain_triggers_for_fraud_prone_types(
    transaction_type: TransactionType,
):
    transaction = make_transaction(
        transaction_type=transaction_type,
        amount=99.0,
        origin_balance=100.0,
    )

    result = FraudRulesEngine().evaluate_all(transaction)[0]

    assert result.rule == "ORIGIN_ACCOUNT_DRAIN"
    assert result.triggered is True
    assert result.severity == "HIGH"
    assert "within 1.0%" in result.description


def test_origin_account_drain_does_not_trigger_outside_tolerance():
    transaction = make_transaction(
        transaction_type="CASH_OUT",
        amount=98.99,
        origin_balance=100.0,
    )

    result = FraudRulesEngine().evaluate_all(transaction)[0]

    assert result.triggered is False
    assert result.severity == "LOW"


@pytest.mark.parametrize(
    ("transaction_type", "amount", "origin_balance"),
    [
        ("PAYMENT", 100.0, 100.0),
        ("CASH_IN", 100.0, 100.0),
        ("CASH_OUT", 0.0, 0.0),
    ],
)
def test_origin_account_drain_excludes_non_drain_cases(
    transaction_type: TransactionType,
    amount: float,
    origin_balance: float,
):
    transaction = make_transaction(
        transaction_type=transaction_type,
        amount=amount,
        origin_balance=origin_balance,
    )

    result = FraudRulesEngine().evaluate_all(transaction)[0]

    assert result.triggered is False


def test_high_value_funded_transfer_triggers_at_threshold():
    transaction = make_transaction(
        transaction_type="TRANSFER",
        amount=1_000_000.0,
        origin_balance=1_000_000.0,
    )

    result = FraudRulesEngine().evaluate_all(transaction)[1]

    assert result.rule == "HIGH_VALUE_FUNDED_TRANSFER"
    assert result.triggered is True
    assert result.severity == "HIGH"
    assert "1,000,000.00" in result.description


@pytest.mark.parametrize(
    ("transaction_type", "amount", "origin_balance"),
    [
        ("TRANSFER", 999_999.99, 1_000_000.0),
        ("TRANSFER", 1_000_000.0, 999_999.99),
        ("CASH_OUT", 1_000_000.0, 1_000_000.0),
    ],
)
def test_high_value_funded_transfer_rejects_non_matching_transactions(
    transaction_type: TransactionType,
    amount: float,
    origin_balance: float,
):
    transaction = make_transaction(
        transaction_type=transaction_type,
        amount=amount,
        origin_balance=origin_balance,
    )

    result = FraudRulesEngine().evaluate_all(transaction)[1]

    assert result.triggered is False
    assert result.severity == "LOW"


def test_custom_config_changes_rule_boundaries():
    engine = FraudRulesEngine(
        RuleConfig(
            drain_relative_tolerance=0.05,
            high_value_transfer_amount=500_000.0,
        )
    )
    transaction = make_transaction(
        transaction_type="TRANSFER",
        amount=500_000.0,
        origin_balance=510_000.0,
    )

    results = engine.evaluate(transaction)

    assert {result.rule for result in results} == {
        "ORIGIN_ACCOUNT_DRAIN",
        "HIGH_VALUE_FUNDED_TRANSFER",
    }


def test_evaluate_all_returns_every_rule_outcome():
    transaction = make_transaction()

    results = FraudRulesEngine().evaluate_all(transaction)

    assert [result.rule for result in results] == [
        "ORIGIN_ACCOUNT_DRAIN",
        "HIGH_VALUE_FUNDED_TRANSFER",
    ]
    assert all(result.triggered is False for result in results)


def test_evaluate_returns_only_triggered_rules():
    transaction = make_transaction(
        transaction_type="CASH_OUT",
        amount=100.0,
        origin_balance=100.0,
    )

    results = FraudRulesEngine().evaluate(transaction)

    assert [result.rule for result in results] == [
        "ORIGIN_ACCOUNT_DRAIN"
    ]
    assert all(result.triggered is True for result in results)


def test_all_fraud_fixture_examples_trigger_at_least_one_rule():
    engine = FraudRulesEngine()
    fraud_transactions = [
        transaction
        for transaction, is_fraud in load_fixture()
        if is_fraud
    ]

    missed = [
        transaction.step
        for transaction in fraud_transactions
        if not engine.evaluate(transaction)
    ]

    assert missed == []
    assert len(fraud_transactions) == 10


def test_legitimate_fixture_examples_do_not_trigger_rules():
    engine = FraudRulesEngine()
    legitimate_transactions = [
        transaction
        for transaction, is_fraud in load_fixture()
        if not is_fraud
    ]

    triggered = [
        transaction.step
        for transaction in legitimate_transactions
        if engine.evaluate(transaction)
    ]

    assert triggered == []
    assert len(legitimate_transactions) == 10


@pytest.mark.parametrize(
    ("step", "expected_rule"),
    [
        (714, "ORIGIN_ACCOUNT_DRAIN"),
        (694, "HIGH_VALUE_FUNDED_TRANSFER"),
    ],
)
def test_fixture_examples_trigger_expected_primary_rule(
    step: int,
    expected_rule: str,
):
    matching_transactions = [
        transaction
        for transaction, is_fraud in load_fixture()
        if is_fraud and transaction.step == step
    ]

    assert len(matching_transactions) == 1
    triggered_rules = FraudRulesEngine().evaluate(
        matching_transactions[0]
    )

    assert expected_rule in {result.rule for result in triggered_rules}
