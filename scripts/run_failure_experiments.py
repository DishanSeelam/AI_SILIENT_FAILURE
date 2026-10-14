"""Run reproducible normal, drift, anomaly, skew and combined monitoring scenarios."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import FEATURE_COLUMNS, REFERENCE_PATH
from monitoring.anomaly import assess_anomaly
from monitoring.drift import assess_drift
from monitoring.health_score import assess_health
from monitoring.skew import assess_skew
from monitoring.validation import validate_record

reference = pd.read_csv(REFERENCE_PATH)
base = reference.sample(250, random_state=42).reset_index(drop=True)


def evaluate(name, current, columns=FEATURE_COLUMNS):
    records = current.to_dict("records")
    validation = [validate_record(row) for row in records]
    valid = [row for row, result in zip(records, validation) if result["status"] == "passed"]
    drift = assess_drift(reference, current)
    skew = assess_skew(columns)
    # A conservative proxy is used to evaluate the monitoring pipeline without calling the prediction API.
    anomalies = sum(row["previous_chargebacks"] >= 2 or row["order_value"] > 3500 for row in valid)
    validation_result = {"status": "passed" if len(valid) == len(records) else "failed"}
    health = assess_health(validation_result, drift, anomalies, skew)
    return {"scenario": name, "health_score": health["health_score"], "severity": health["severity"],
            "failure_types": health["failure_types"], "drift_status": drift["status"],
            "drifted_features": drift["drifted_features"], "validation_status": validation_result["status"],
            "anomaly_count": anomalies, "skew_status": skew["status"],
            "recommended_action": health["recommended_action"]}


drifted = base.copy()
drifted["order_value"] *= 3.2; drifted["unit_price"] *= 2.6; drifted["customer_age"] = (drifted["customer_age"] + 25).clip(18, 100)
anomalous = base.copy(); anomalous.loc[:29, "previous_chargebacks"] = 3; anomalous.loc[:29, "order_value"] = 6000
combined = drifted.copy(); combined.loc[:29, "previous_chargebacks"] = 3; combined.loc[:29, "order_value"] = 6500
results = [evaluate("A_normal", base), evaluate("B_data_drift", drifted), evaluate("C_anomaly", anomalous),
           evaluate("D_training_serving_skew", base, FEATURE_COLUMNS[:-1]), evaluate("E_combined", combined, FEATURE_COLUMNS[:-1])]
output = ROOT / "logs" / "controlled_failure_experiments.json"
output.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
