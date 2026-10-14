"""Load a selected MLflow Registry model, retaining joblib as an offline fallback."""
import json
from functools import lru_cache
import joblib
from config.settings import MANIFEST_PATH, MLFLOW_TRACKING_URI, MODEL_PATH


@lru_cache(maxsize=1)
def load_serving_model():
    metadata = {"model_version": "local", "model_stage": "fallback", "fraud_threshold": .5}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            return mlflow.sklearn.load_model(manifest["model_uri"]), manifest
        except Exception as error:
            metadata["registry_load_warning"] = str(error)
    return joblib.load(MODEL_PATH), metadata
