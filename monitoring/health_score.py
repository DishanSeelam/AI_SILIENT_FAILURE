"""Explainable multi-signal pipeline-health assessment."""

def assess_health(validation, drift, anomaly_count, skew):
    score = 100
    causes = []
    if validation["status"] == "failed":
        score -= 30; causes.append("feature validation")
    if drift["status"] == "drift_detected":
        score -= min(35, 10 + 5 * drift["drifted_features"]); causes.append("data drift")
    if anomaly_count:
        score -= min(25, 10 + anomaly_count); causes.append("transaction anomalies")
    if skew["status"] == "skew_detected":
        score -= 25; causes.append("training-serving skew")
    score = max(0, score)
    severity = "healthy" if score >= 85 else "warning" if score >= 60 else "critical"
    action = ("Continue monitoring." if not causes else
              "Investigate " + ", ".join(causes) + "; validate data and review the deployed model.")
    return {"health_score": score, "severity": severity, "failure_types": causes or ["none"],
            "recommended_action": action}
