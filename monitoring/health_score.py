"""Unified health scoring and remediation recommendations."""


def calculate_health_score(signals: dict) -> dict:
    """Combine independent monitoring signals into one actionable assessment.

    The score starts at 100. The documented deductions are: validation 30,
    drift up to 30, prediction anomaly 20, and training-serving skew 20.
    """
    score = 100
    failure_types: list[str] = []
    recommendations: list[str] = []

    if not signals["validation_passed"]:
        score -= 30
        failure_types.append("feature_validation_failure")
        recommendations.append("Fix the upstream input schema or ETL pipeline before inference.")

    if signals["drift_detected"]:
        drift_penalty = min(30, 10 + 5 * signals["drifted_feature_count"])
        score -= drift_penalty
        failure_types.append("data_drift")
        recommendations.append("Review shifted features and retrain the model if the change persists.")

    if signals["anomaly_detected"]:
        score -= 20
        failure_types.append("prediction_anomaly")
        recommendations.append("Manually review the anomalous transaction and inspect its feature values.")

    if signals["skew_detected"]:
        score -= 20
        failure_types.append("training_serving_skew")
        recommendations.append("Synchronize the serving preprocessing pipeline with the training contract.")

    score = max(0, score)
    if score >= 85:
        severity, status = "normal", "healthy"
    elif score >= 60:
        severity, status = "warning", "attention_required"
    else:
        severity, status = "critical", "intervention_required"

    return {
        "health_score": score,
        "status": status,
        "severity": severity,
        "failure_types": failure_types,
        "recommended_actions": recommendations or ["Continue routine monitoring."],
        "weights": {
            "feature_validation": 30,
            "data_drift": "10 + 5 per drifted feature, capped at 30",
            "prediction_anomaly": 20,
            "training_serving_skew": 20,
        },
    }
