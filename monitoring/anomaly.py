"""Prediction-anomaly detection using Isolation Forest."""

import pandas as pd


ANOMALY_FEATURES = [
    "fraud_probability",
    "order_value",
    "customer_tenure_days",
    "hour_of_day",
    "previous_chargebacks",
    "shipping_distance_km",
]


def build_anomaly_input(transaction: dict, fraud_probability: float) -> pd.DataFrame:
    """Create the fixed feature vector used by the fitted anomaly detector."""
    row = {column: transaction[column] for column in ANOMALY_FEATURES if column != "fraud_probability"}
    row["fraud_probability"] = fraud_probability
    return pd.DataFrame([[row[column] for column in ANOMALY_FEATURES]], columns=ANOMALY_FEATURES)


def assess_prediction_anomaly(model: object, transaction: dict, fraud_probability: float) -> dict:
    """Return Isolation Forest anomaly status and its signed decision score."""
    anomaly_input = build_anomaly_input(transaction, fraud_probability)
    score = float(model.decision_function(anomaly_input)[0])
    is_anomaly = bool(model.predict(anomaly_input)[0] == -1)
    return {"anomaly_detected": is_anomaly, "anomaly_score": round(score, 4)}
