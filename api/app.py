"""FastAPI application for e-commerce fraud prediction."""

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import DriftRequest, HealthRequest, PredictionResponse, SkewRequest, TransactionRequest
from config.settings import settings
from database.event_store import log_health, log_prediction, recent_events
from monitoring.drift import assess_drift
from monitoring.anomaly import assess_prediction_anomaly
from monitoring.explainability import explain_prediction
from monitoring.skew import assess_skew
from monitoring.health_score import calculate_health_score
from monitoring.validation import validate_features
from utils.features import prepare_features

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Provide a friendly entry point for people opening the base API URL."""
    return {
        "message": "AI Silent Failure Detection API is running.",
        "documentation": "/docs",
        "health_check": "/health",
        "training_metrics": "/metrics",
    }


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return service and trained-model availability."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "model_status": "available" if settings.model_path.exists() else "not_trained",
    }


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    """Load the production model bundle once per running API process."""
    if not settings.model_path.exists():
        raise FileNotFoundError("No trained model artifact found.")
    bundle = joblib.load(settings.model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "alert_threshold" not in bundle:
        raise ValueError("The saved model artifact has an unexpected format.")
    return bundle


@lru_cache(maxsize=1)
def load_anomaly_detector() -> Any:
    """Load the fitted Isolation Forest once per running API process."""
    if not settings.anomaly_model_path.exists():
        raise FileNotFoundError("No anomaly detector artifact found.")
    return joblib.load(settings.anomaly_model_path)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(transaction: TransactionRequest) -> PredictionResponse:
    """Return a fraud probability and alert decision for one validated transaction."""
    transaction_data = transaction.model_dump()
    try:
        model_bundle = load_model_bundle()
        validation_errors = validate_features(transaction_data)
        anomaly_detector = load_anomaly_detector()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {error}") from error
    if validation_errors:
        raise HTTPException(status_code=422, detail={"validation_errors": validation_errors})

    features = prepare_features(pd.DataFrame([transaction_data]))
    probability = float(model_bundle["pipeline"].predict_proba(features)[0, 1])
    anomaly_result = assess_prediction_anomaly(anomaly_detector, transaction_data, probability)
    threshold = float(model_bundle["alert_threshold"])
    predicted_label = int(probability >= threshold)
    should_explain = bool(predicted_label or anomaly_result["anomaly_detected"])
    contributions: list[dict] = []
    if should_explain:
        try:
            contributions = explain_prediction(model_bundle["pipeline"], features)
        except ImportError:
            should_explain = False
    response = PredictionResponse(
        fraud_probability=round(probability, 4),
        alert_threshold=threshold,
        predicted_label=predicted_label,
        decision="fraud_alert" if predicted_label else "normal_transaction",
        confidence=round(max(probability, 1 - probability), 4),
        validation_status="passed",
        anomaly_detected=anomaly_result["anomaly_detected"],
        anomaly_score=anomaly_result["anomaly_score"],
        explanation_generated=should_explain,
        top_feature_contributions=contributions,
    )
    log_prediction(response.model_dump())
    return response


@app.get("/metrics", tags=["monitoring"])
def metrics() -> dict[str, Any]:
    """Return the latest saved offline training metrics."""
    metrics_path = settings.model_path.with_name("training_metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Train the model before requesting metrics.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/monitor/drift", tags=["monitoring"])
def monitor_drift(request: DriftRequest) -> dict[str, Any]:
    """Check a batch of recent transactions for distribution drift."""
    if not settings.reference_data_path.exists():
        raise HTTPException(status_code=503, detail="Drift reference data missing. Retrain the model first.")
    reference_data = pd.read_csv(settings.reference_data_path)
    current_raw_data = pd.DataFrame([transaction.model_dump() for transaction in request.transactions])
    current_features = prepare_features(current_raw_data)
    return assess_drift(reference_data, current_features)


@app.post("/monitor/skew", tags=["monitoring"])
def monitor_skew(request: SkewRequest) -> dict[str, Any]:
    """Validate a serving feature contract against the training contract."""
    if not settings.training_contract_path.exists():
        raise HTTPException(status_code=503, detail="Training contract missing. Retrain the model first.")
    training_contract = json.loads(settings.training_contract_path.read_text(encoding="utf-8"))
    return assess_skew(training_contract, request.model_dump())


@app.post("/monitor/health", tags=["monitoring"])
def monitor_health(request: HealthRequest) -> dict[str, Any]:
    """Combine validation, drift, anomaly, and skew findings into one health score."""
    result = calculate_health_score(request.model_dump())
    log_health(result)
    return result


@app.get("/monitor/history", tags=["monitoring"])
def monitor_history() -> dict[str, list[dict[str, Any]]]:
    """Return recent persisted prediction and unified-health events."""
    predictions, health = recent_events()
    return {"predictions": predictions.to_dict(orient="records"), "health": health.to_dict(orient="records")}
