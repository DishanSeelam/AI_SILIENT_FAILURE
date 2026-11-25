"""Dependency-free smoke tests for the API module."""

from api.app import health


def test_health_endpoint() -> None:
    result = health()
    assert result["status"] in {"ready", "model_not_trained"}
    assert "status" in result
