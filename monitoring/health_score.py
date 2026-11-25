"""Explainable multi-signal pipeline-health assessment."""

def assess_health(validation, drift, anomaly_count, skew):
    score = 100
    causes = []
    if validation["status"] == "failed":
        score -= 30; causes.append("feature validation")
    if drift["status"] == "drift_detected":
        score -= min(30, 10 + 5 * drift["drifted_features"]); causes.append("data drift")
    if anomaly_count:
        score -= 20; causes.append("transaction anomalies")
    if skew["status"] == "skew_detected":
        score -= 20; causes.append("training-serving skew")
    score = max(0, score)
    severity = "healthy" if score >= 85 else "warning" if score >= 60 else "critical"
    action = ("Continue monitoring." if not causes else
              "Investigate " + ", ".join(causes) + "; validate data and review the deployed model.")
    status = "healthy" if severity == "healthy" else "attention_required" if severity == "warning" else "intervention_required"
    return {"health_score": score, "status": status, "severity": severity,
            "failure_types": causes or ["none"], "recommended_action": action}


def calculate_health_score(signals: dict):
    """Compatibility wrapper for the health endpoint and legacy test/report code."""
    return assess_health(
        {"status": "passed" if signals["validation_passed"] else "failed"},
        {"status": "drift_detected" if signals["drift_detected"] else "stable",
         "drifted_features": signals["drifted_feature_count"]},
        int(signals["anomaly_detected"]),
        {"status": "skew_detected" if signals["skew_detected"] else "consistent"},
    )
