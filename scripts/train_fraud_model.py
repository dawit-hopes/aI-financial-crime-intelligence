"""Train and evaluate the PaySim XGBoost fraud model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.features import FraudFeatureBuilder
from src.paysim_data import (
    DATASET,
    PRESERVED_DECISION_THRESHOLD,
    RANDOM_STATE,
    TEST_MIN_STEP,
    THRESHOLD_SELECTION,
    THRESHOLD_SELECTION_DATASET,
    TRAIN_MAX_STEP,
    VALIDATION_MAX_STEP,
    VALIDATION_MIN_STEP,
    add_past_only_history,
    assert_fraud_feature_parity,
    build_fraud_feature_frame,
    load_data,
    resolve_data_path,
)

DEFAULT_MODEL_PATH = Path("src/artifacts/fraud_model.json")
DEFAULT_CONFIG_PATH = Path("src/artifacts/fraud_model_config.json")
MODEL_VERSION = "1.0.0"
THRESHOLD_CANDIDATES = sorted(
    {
        round(value, 2)
        for value in np.arange(0.05, 1.00, 0.05)
    }
    | {PRESERVED_DECISION_THRESHOLD}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "threshold": threshold,
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def evaluate_split(
    split_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    metrics = threshold_metrics(labels, probabilities, threshold)
    metrics["pr_auc"] = float(
        average_precision_score(labels, probabilities)
    )

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"Threshold: {threshold:.2f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(
        "Legitimate false-positive rate: "
        f"{metrics['false_positive_rate']:.4%}"
    )
    print(
        f"TN={metrics['true_negatives']:,} "
        f"FP={metrics['false_positives']:,} "
        f"FN={metrics['false_negatives']:,} "
        f"TP={metrics['true_positives']:,}"
    )
    return metrics


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    print(f"Loading PaySim from {data_path}")
    data = add_past_only_history(load_data(data_path))

    train = data.loc[data["step"] <= TRAIN_MAX_STEP]
    validation = data.loc[
        data["step"].between(VALIDATION_MIN_STEP, VALIDATION_MAX_STEP)
    ]
    test = data.loc[data["step"] >= TEST_MIN_STEP]

    train_features = build_fraud_feature_frame(train)
    validation_features = build_fraud_feature_frame(validation)
    assert_fraud_feature_parity(train, train_features)
    assert_fraud_feature_parity(validation, validation_features)

    y_train = train["isFraud"].to_numpy()
    y_validation = validation["isFraud"].to_numpy()
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    print(
        f"Training on {len(train):,} rows "
        f"({positive:,} fraud, scale_pos_weight={scale_pos_weight:.2f})"
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        train_features,
        y_train,
        eval_set=[(validation_features, y_validation)],
        verbose=50,
    )

    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    threshold_candidates = [
        threshold_metrics(y_validation, validation_probabilities, threshold)
        for threshold in THRESHOLD_CANDIDATES
    ]
    print("\nValidation threshold candidates:")
    candidate_frame = pd.DataFrame(threshold_candidates)
    print(
        candidate_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    decision_threshold = PRESERVED_DECISION_THRESHOLD
    print(
        f"\nSelected threshold: {decision_threshold:.2f} "
        f"({THRESHOLD_SELECTION})"
    )

    validation_metrics = evaluate_split(
        "VALIDATION",
        y_validation,
        validation_probabilities,
        decision_threshold,
    )

    test_features = build_fraud_feature_frame(test)
    assert_fraud_feature_parity(test, test_features)
    test_probabilities = model.predict_proba(test_features)[:, 1]
    test_metrics = evaluate_split(
        "TEST",
        test["isFraud"].to_numpy(),
        test_probabilities,
        decision_threshold,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_path)

    artifact = {
        "model_version": MODEL_VERSION,
        "decision_threshold": decision_threshold,
        "threshold_selection": THRESHOLD_SELECTION,
        "threshold_selection_dataset": THRESHOLD_SELECTION_DATASET,
        "threshold_candidates": threshold_candidates,
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float(scale_pos_weight),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        },
        "feature_names": FraudFeatureBuilder.FEATURE_NAMES,
        "training_metadata": {
            "dataset": DATASET,
            "train_max_step": TRAIN_MAX_STEP,
            "validation_steps": [VALIDATION_MIN_STEP, VALIDATION_MAX_STEP],
            "test_min_step": TEST_MIN_STEP,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    args.config_path.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved model to {args.model_path}")
    print(f"Saved model config to {args.config_path}")


if __name__ == "__main__":
    main()
