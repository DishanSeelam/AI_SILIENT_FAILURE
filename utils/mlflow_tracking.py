"""Local MLflow configuration and model-registry helpers."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import PROJECT_ROOT, settings


MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "model_manifest.json"


def configure_mlflow() -> object:
    """Configure a project-local MLflow tracking store and return the module."""
    import mlflow

    (PROJECT_ROOT / "database").mkdir(exist_ok=True)
    artifact_root = PROJECT_ROOT / "mlartifacts" / "fraud_detection_v4"
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            settings.mlflow_experiment_name, artifact_location=f"file:///{artifact_root.as_posix()}"
        )
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
    return mlflow


def write_model_manifest(payload: dict) -> None:
    """Persist the active registered-model metadata for API model provenance."""
    MODEL_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_model_manifest() -> dict:
    """Return local model provenance, or a fallback marker before registration."""
    if not MODEL_MANIFEST_PATH.exists():
        return {"model_version": "local", "model_stage": "Fallback", "model_uri": ""}
    return json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
