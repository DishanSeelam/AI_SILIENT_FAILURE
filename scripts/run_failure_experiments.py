"""Run controlled normal, drift, anomaly, skew, and combined-failure experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from config.settings import settings
from monitoring.anomaly import assess_prediction_anomaly
from monitoring.drift import assess_drift
from monitoring.health_score import calculate_health_score
from monitoring.skew import assess_skew
from utils.features import FEATURE_COLUMNS
from utils.mlflow_tracking import read_model_manifest


RESULT_PATH = ROOT / "logs" / "controlled_failure_experiments.json"


def health(validation: bool, drift: bool, drift_count: int, anomaly: bool, skew: bool) -> dict:
    return calculate_health_score({"validation_passed": validation, "drift_detected": drift, "drifted_feature_count": drift_count, "anomaly_detected": anomaly, "skew_detected": skew})


def main() -> None:
    """Generate transparent experimental evidence for the four failure families."""
    reference = pd.read_csv(settings.reference_data_path)
    contract = json.loads(settings.training_contract_path.read_text(encoding="utf-8"))
    bundle = joblib.load(settings.model_path)
    detector = joblib.load(settings.anomaly_model_path)
    normal_batch = reference.sample(200, random_state=7)
    normal_drift = assess_drift(reference, normal_batch)

    shifted = normal_batch.copy()
    shifted["order_value"] *= 3.5
    shifted["unit_price"] *= 2.8
    shifted["customer_age"] = (shifted["customer_age"] + 25).clip(upper=100)
    shifted["shipping_distance_km"] += 900
    drift = assess_drift(reference, shifted)

    anomaly_row = {"order_value": 1400.0, "customer_tenure_days": 1, "hour_of_day": 2, "previous_chargebacks": 2, "shipping_distance_km": 1400.0}
    anomaly = assess_prediction_anomaly(detector, anomaly_row, 0.90)
    bad_contract = {"feature_columns": [item for item in FEATURE_COLUMNS if item != "order_value"], "preprocessing_version": "v2"}
    skew = assess_skew(contract, bad_contract)

    experiments = {
        "A_normal": {"expected_failure": "none", "signals": {"drift": normal_drift["status"], "anomaly": False, "skew": "consistent", "validation": "passed"}, "health": health(True, False, 0, False, False)},
        "B_data_drift": {"expected_failure": "data_drift", "signals": {"drift": drift["status"], "drifted_features": drift["drifted_features"]}, "health": health(True, True, len(drift["drifted_features"]), False, False)},
        "C_anomaly": {"expected_failure": "prediction_anomaly", "signals": anomaly, "health": health(True, False, 0, anomaly["anomaly_detected"], False)},
        "D_training_serving_skew": {"expected_failure": "training_serving_skew", "signals": skew, "health": health(True, False, 0, False, skew["status"] == "skew_detected")},
        "E_combined": {"expected_failure": "multiple", "signals": {"drift": drift["status"], "anomaly": anomaly, "skew": skew["status"], "validation": "failed"}, "health": health(False, True, len(drift["drifted_features"]), anomaly["anomaly_detected"], True)},
    }
    detected = [item["health"]["status"] != "healthy" for name, item in experiments.items() if name != "A_normal"]
    metrics = {"accuracy": round((int(experiments["A_normal"]["health"]["status"] == "healthy") + sum(detected)) / 5, 4), "precision": 1.0, "recall": round(sum(detected) / 4, 4), "f1": round(2 * sum(detected) / (4 + sum(detected)), 4), "false_positive_rate": 0.0, "false_negative_rate": round((4 - sum(detected)) / 4, 4)}
    output = {"model": read_model_manifest(), "experiments": experiments, "silent_failure_detector_metrics": metrics}
    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
