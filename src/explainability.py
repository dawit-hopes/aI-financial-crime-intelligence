class Explainability:
    """
    Generate explanations for fraud predictions using
    XGBoost's native feature contributions.
    """

    def __init__(self, model, threshold=0.92):
        self.model = model
        self.threshold = threshold

    def explain_transaction(
        self,
        features: dict[str, float],
        top_n: int = 5,
    ):
        fraud_probability = self.model.predict_proba(features)
        prediction = "FRAUD" if fraud_probability >= self.threshold else "LEGITIMATE"

        contributions = self.model.feature_contributions(features)

        explanation = []

        for feature, value, contribution in zip(
            features.keys(),
            features.values(),
            contributions,
            strict=True,
        ):
            contribution = float(contribution)
            value = float(value)

            # Only keep features that support the prediction.
            if prediction == "FRAUD" and contribution <= 0:
                continue

            if prediction == "LEGITIMATE" and contribution >= 0:
                continue

            # Only show the active transaction type.
            if feature.startswith("type_") and value != 1:
                continue

            explanation.append(
                {
                    "feature": feature,
                    "value": value,
                    "contribution": contribution,
                    "abs_contribution": abs(contribution),
                }
            )

        # Most influential features first.
        explanation.sort(
            key=lambda item: item["abs_contribution"],
            reverse=True,
        )

        explanation = explanation[:top_n]

        reasons = []

        for item in explanation:
            description = self.describe_shap_reason(
                feature=item["feature"],
                value=item["value"],
                shap_value=item["contribution"],
            )

            if description is None:
                continue

            direction = "FRAUD" if item["contribution"] > 0 else "LEGITIMATE"

            reasons.append(
                {
                    "feature": item["feature"],
                    "value": item["value"],
                    "contribution": item["contribution"],
                    "direction": direction,
                    "description": description,
                }
            )

        return {
            "fraud_probability": fraud_probability,
            "prediction": prediction,
            "reasons": reasons,
        }

    def describe_shap_reason(
        self,
        feature,
        value,
        shap_value,
    ):
        if shap_value == 0:
            return None

        direction = "increased" if shap_value > 0 else "reduced"

        if feature == "amount":
            return (
                f"The transaction amount of {value:,.2f} "
                f"{direction} the model's fraud risk."
            )

        if feature == "oldbalanceOrg":
            return (
                f"The origin account balance of {value:,.2f} "
                f"{direction} the model's fraud risk."
            )

        if feature == "oldbalanceDest":
            return (
                f"The destination account balance of {value:,.2f} "
                f"{direction} the model's fraud risk."
            )

        if feature == "hour_of_day":
            return (
                f"The transaction occurred around hour {int(value)}, "
                f"which {direction} the model's fraud risk."
            )

        if feature == "step":
            return (
                f"The transaction occurred at step {int(value)}, "
                f"which {direction} the model's fraud risk."
            )

        if feature.startswith("type_"):
            if value != 1:
                return None

            transaction_type = feature.replace("type_", "").replace("_", " ")

            return (
                f"The transaction type was {transaction_type}, "
                f"which {direction} the model's fraud risk."
            )

        if feature == "dest_time_since_previous":
            if value == -1:
                return (
                    "No previous destination transaction history was available, "
                    f"which {direction} the model's fraud risk."
                )

            return (
                f"The previous destination transaction was "
                f"{value:.0f} steps earlier, which {direction} the model's fraud risk."
            )

        if feature == "dest_previous_transaction_count":
            return (
                f"The destination account had {int(value)} previous transactions, "
                f"which {direction} the model's fraud risk."
            )

        if feature == "dest_previous_total_amount":
            return (
                f"The destination account's historical transaction volume was "
                f"{value:,.2f}, which {direction} the model's fraud risk."
            )

        if feature == "dest_previous_avg_amount":
            return (
                f"The destination account's historical average transaction was "
                f"{value:,.2f}, which {direction} the model's fraud risk."
            )

        if feature == "orig_time_since_previous":
            if value == -1:
                return (
                    "No previous origin transaction history was available, "
                    f"which {direction} the model's fraud risk."
                )

            return (
                f"The previous origin transaction was "
                f"{value:.0f} steps earlier, which {direction} the model's fraud risk."
            )

        if feature == "orig_previous_transaction_count":
            return (
                f"The origin account had {int(value)} previous transactions, "
                f"which {direction} the model's fraud risk."
            )

        if feature == "orig_previous_total_amount":
            return (
                f"The origin account's historical transaction volume was "
                f"{value:,.2f}, which {direction} the model's fraud risk."
            )

        if feature == "orig_previous_avg_amount":
            return (
                f"The origin account's historical average transaction was "
                f"{value:,.2f}, which {direction} the model's fraud risk."
            )

        return None
