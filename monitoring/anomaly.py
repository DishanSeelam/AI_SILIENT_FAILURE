"""Prediction-aware transaction anomaly detection."""
import joblib
import pandas as pd
from config.settings import ANOMALY_PATH


def assess_anomaly(record: dict, fraud_probability: float):
    if not ANOMALY_PATH.exists():
        return {"status": "unavailable", "anomaly_detected": False, "anomaly_score": None}
    saved = joblib.load(ANOMALY_PATH)
    features = saved["features"]
    row = {**record, "fraud_probability": fraud_probability}
    frame = pd.DataFrame([{key: row[key] for key in features}])
    score = float(saved["model"].decision_function(frame)[0])
    return {"status": "anomaly" if score < 0 else "normal", "anomaly_detected": score < 0,
            "anomaly_score": round(score, 4)}
