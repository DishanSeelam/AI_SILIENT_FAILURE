"""FastAPI service for fraud prediction and silent-failure monitoring."""
import json
import sys
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.schemas import DriftRequest, HealthRequest, MonitorRequest, SkewRequest, TransactionRequest
from config.settings import FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, REFERENCE_PATH
from database.event_store import log_event, recent
from monitoring.anomaly import assess_anomaly
from monitoring.drift import assess_drift
from monitoring.health_score import assess_health
from monitoring.skew import assess_skew
from monitoring.validation import validate_record
from utils.model_loader import load_serving_model

app = FastAPI(title="AI Silent Failure Detection", version="1.0.0")


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model unavailable. Generate data and train models first.")
    return load_serving_model()


def record(transaction: TransactionRequest):
    data = transaction.model_dump()
    data["transaction_timestamp"] = data["transaction_timestamp"].isoformat()
    return data


@app.get("/")
def root():
    return {"service": "AI Silent Failure Detection", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ready" if MODEL_PATH.exists() else "model_not_trained"}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics found. Train a model first.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(transaction: TransactionRequest):
    raw = record(transaction)
    validation = validate_record(raw)
    if validation["status"] == "failed":
        raise HTTPException(422, {"validation": validation})
    model, model_info = load_model()
    probability = float(model.predict_proba(pd.DataFrame([raw])[FEATURE_COLUMNS])[:, 1][0])
    metrics_data = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {"threshold": .5}
    threshold = model_info.get("fraud_threshold", metrics_data.get("threshold", .5))
    anomaly = assess_anomaly(raw, probability)
    response = {"fraud_probability": round(probability, 4), "alert_threshold": threshold,
                "predicted_label": int(probability >= threshold),
                "decision": "fraud_alert" if probability >= threshold else "approved",
                "validation_status": validation["status"], **anomaly,
                "model_version": model_info.get("model_version", "local"), "model_stage": model_info.get("model_stage", "fallback")}
    log_event("prediction", response)
    return response


@app.post("/monitor/assess")
def monitor_assess(request: MonitorRequest):
    if len(request.transactions) < 10:
        raise HTTPException(422, "Provide at least 10 transactions for a meaningful drift assessment.")
    raw = [record(item) for item in request.transactions]
    validations = [validate_record(item) for item in raw]
    valid = [item for item, result in zip(raw, validations) if result["status"] == "passed"]
    if not valid:
        raise HTTPException(422, "All provided transactions failed feature validation.")
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.DataFrame(valid)
    drift = assess_drift(reference, current)
    skew = assess_skew(request.serving_columns or FEATURE_COLUMNS, request.preprocessing_version)
    model, model_info = load_model()
    probabilities = model.predict_proba(current[FEATURE_COLUMNS])[:, 1]
    anomalies = [assess_anomaly(item, float(probability)) for item, probability in zip(valid, probabilities)]
    validation_summary = {"status": "passed" if len(valid) == len(raw) else "failed",
                          "invalid_records": len(raw) - len(valid),
                          "issues": [v["issues"] for v in validations if v["status"] == "failed"]}
    health = assess_health(validation_summary, drift, sum(a["anomaly_detected"] for a in anomalies), skew)
    response = {"batch_size": len(raw), "valid_records": len(valid), "validation": validation_summary,
                "drift": drift, "anomaly": {"anomaly_count": sum(a["anomaly_detected"] for a in anomalies),
                                               "sample_scores": anomalies[:10]},
                "skew": skew, "health": health, "model_version": model_info.get("model_version", "local"),
                "model_stage": model_info.get("model_stage", "fallback")}
    log_event("monitoring_assessment", response)
    return response


@app.post("/monitor/drift")
def monitor_drift(request: DriftRequest):
    """Compatibility endpoint: assess numerical and categorical data drift in a batch."""
    rows = [record(item) for item in request.transactions]
    validations = [validate_record(row) for row in rows]
    errors = [issue for validation in validations for issue in validation.get("issues", [])]
    if errors:
        raise HTTPException(422, {"validation_errors": errors})
    report = assess_drift(pd.read_csv(REFERENCE_PATH), pd.DataFrame(rows))
    log_event("drift", report)
    return report


@app.post("/monitor/anomaly")
def monitor_anomaly(request: DriftRequest):
    """Score a batch for unusual input/prediction combinations."""
    rows = [record(item) for item in request.transactions]
    model, model_info = load_model()
    probabilities = model.predict_proba(pd.DataFrame(rows)[FEATURE_COLUMNS])[:, 1]
    scores = [assess_anomaly(row, float(probability)) for row, probability in zip(rows, probabilities)]
    result = {"status": "anomalies_detected" if any(score["anomaly_detected"] for score in scores) else "normal",
              "anomaly_count": sum(score["anomaly_detected"] for score in scores),
              "batch_size": len(rows), "model_version": model_info.get("model_version", "local"),
              "samples": scores[:20]}
    log_event("anomaly", result)
    return result


@app.post("/monitor/skew")
def monitor_skew(request: SkewRequest):
    result = assess_skew(request.serving_columns, request.preprocessing_version)
    log_event("skew", result)
    return result


@app.post("/monitor/health")
def monitor_health(request: HealthRequest):
    validation = {"status": "passed" if request.validation_passed else "failed"}
    drift = {"status": "drift_detected" if request.drift_detected else "stable",
             "drifted_features": request.drifted_feature_count}
    skew = {"status": "skew_detected" if request.skew_detected else "consistent"}
    result = assess_health(validation, drift, int(request.anomaly_detected), skew)
    log_event("health", result)
    return result


@app.get("/monitor/history")
def monitor_history(limit: int = 25):
    return recent(min(max(limit, 1), 100))
