from monitoring.health_score import assess_health


def test_clean_pipeline_is_healthy():
    result = assess_health({"status": "passed"}, {"status": "stable", "drifted_features": 0}, 0,
                           {"status": "consistent"})
    assert result["health_score"] == 100
    assert result["severity"] == "healthy"


def test_combined_failures_are_critical():
    result = assess_health({"status": "failed"}, {"status": "drift_detected", "drifted_features": 5}, 10,
                           {"status": "skew_detected"})
    assert result["health_score"] < 60
    assert result["severity"] == "critical"
