"""Tests for the project's unified health-score contribution."""

from monitoring.health_score import calculate_health_score


def test_healthy_pipeline_scores_100() -> None:
    result = calculate_health_score(
        {
            "validation_passed": True,
            "drift_detected": False,
            "drifted_feature_count": 0,
            "anomaly_detected": False,
            "skew_detected": False,
        }
    )
    assert result["health_score"] == 100
    assert result["status"] == "healthy"


def test_multiple_failures_are_critical() -> None:
    result = calculate_health_score(
        {
            "validation_passed": False,
            "drift_detected": True,
            "drifted_feature_count": 4,
            "anomaly_detected": True,
            "skew_detected": True,
        }
    )
    assert result["health_score"] == 0
    assert result["severity"] == "critical"
