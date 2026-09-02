import pandas as pd
import shap


class Explainability:
    """
    Generate SHAP-based explanations for fraud predictions.
    """

    def __init__(self, model, explainer, threshold=0.92):
        self.model = model
        self.explainer = explainer
        self.threshold = threshold

    def explain_transaction(
        self,
        features: dict[str, float],
        top_n: int = 5,
    ):
        features_df = pd.DataFrame([features])

        fraud_probability = float(self.model.predict_proba(features_df)[0, 1])

        prediction = "FRAUD" if fraud_probability >= self.threshold else "LEGITIMATE"

        shap_values = self.explainer.shap_values(features_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        explanation = pd.DataFrame(
            {
                "feature": features_df.columns,
                "value": features_df.iloc[0].values,
                "shap_value": shap_values,
            }
        )

        if prediction == "FRAUD":
            explanation = explanation[explanation["shap_value"] > 0].copy()

        else:
            explanation = explanation[explanation["shap_value"] < 0].copy()

        # Only show the active transaction type.
        is_categorical = explanation["feature"].str.startswith("type_")

        explanation = explanation[
            (~is_categorical) | (explanation["value"] == 1)
        ].copy()

        explanation["abs_shap"] = explanation["shap_value"].abs()
        explanation["direction"] = explanation["shap_value"].apply(
            lambda value: "FRAUD" if value > 0 else "LEGITIMATE"
        )

        explanation = (
            explanation.sort_values("abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        reasons = []

        for _, row in explanation.iterrows():
            description = self.describe_shap_reason(
                feature=row["feature"],
                value=row["value"],
                shap_value=row["shap_value"],
            )

            if description is None:
                continue

            reasons.append(
                {
                    "feature": row["feature"],
                    "value": float(row["value"]),
                    "contribution": float(row["shap_value"]),
                    "direction": row["direction"],
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
        if shap_value > 0:
            direction = "toward the model's fraud prediction"
        elif shap_value < 0:
            direction = "toward the model's legitimate prediction"
        else:
            return None

        if feature == "amount":
            return f"Transaction amount ({value:,.2f}) " f"contributed {direction}."

        if feature == "oldbalanceOrg":
            return f"Origin account balance ({value:,.2f}) " f"contributed {direction}."

        if feature == "oldbalanceDest":
            return (
                f"Destination account balance ({value:,.2f}) "
                f"contributed {direction}."
            )

        if feature == "hour_of_day":
            return (
                f"Transaction timing around hour {int(value)} "
                f"contributed {direction}."
            )

        if feature == "step":
            return (
                f"Transaction timing (step {int(value)}) " f"contributed {direction}."
            )

        if feature.startswith("type_"):
            if value != 1:
                return None

            transaction_type = feature.replace("type_", "").replace("_", " ")

            return (
                f"Transaction type is {transaction_type}, "
                f"which contributed {direction}."
            )

        if feature == "dest_time_since_previous":
            if value == -1:
                return (
                    "No previous destination transaction history "
                    f"was available, which contributed {direction}."
                )

            return (
                f"Time since the previous destination transaction "
                f"was {value:.0f} steps, which contributed {direction}."
            )

        if feature == "dest_previous_transaction_count":
            return (
                f"Destination account has {int(value)} previous "
                f"transactions, which contributed {direction}."
            )

        if feature == "dest_previous_total_amount":
            return (
                f"Destination account historical transaction volume "
                f"was {value:,.2f}, which contributed {direction}."
            )

        if feature == "dest_previous_avg_amount":
            return (
                f"Destination account historical average transaction "
                f"was {value:,.2f}, which contributed {direction}."
            )

        if feature == "orig_time_since_previous":
            if value == -1:
                return (
                    "No previous origin transaction history "
                    f"was available, which contributed {direction}."
                )

            return (
                f"Time since the previous origin transaction "
                f"was {value:.0f} steps, which contributed {direction}."
            )

        if feature == "orig_previous_transaction_count":
            return (
                f"Origin account has {int(value)} previous "
                f"transactions, which contributed {direction}."
            )

        if feature == "orig_previous_total_amount":
            return (
                f"Origin account historical transaction volume "
                f"was {value:,.2f}, which contributed {direction}."
            )

        if feature == "orig_previous_avg_amount":
            return (
                f"Origin account historical average transaction "
                f"was {value:,.2f}, which contributed {direction}."
            )

        return None
