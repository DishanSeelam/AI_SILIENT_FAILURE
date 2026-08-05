"""Smoke tests for API scaffold."""

from fastapi.testclient import TestClient

from api.app import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_requires_trained_model() -> None:
    response = TestClient(app).post("/predict", json={})
    assert response.status_code in {422, 503}
