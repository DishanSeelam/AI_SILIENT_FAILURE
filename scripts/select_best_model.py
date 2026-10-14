"""Register the strongest completed MLflow candidate by validation-aligned F2 score."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import MANIFEST_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


def main():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise RuntimeError("No MLflow experiment found. Run scripts/run_mlflow_experiments.py first.")
    runs = client.search_runs([experiment.experiment_id], order_by=["metrics.f2 DESC"], max_results=50)
    completed = [run for run in runs if "f2" in run.data.metrics]
    if not completed:
        raise RuntimeError("No completed model runs with an F2 metric were found.")
    best = completed[0]
    model_uri = f"runs:/{best.info.run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    manifest = {"model_uri": f"models:/{REGISTERED_MODEL_NAME}/{registered.version}",
                "model_version": str(registered.version), "model_stage": "candidate",
                "source_run_id": best.info.run_id, "selection_metric": "f2",
                "selection_value": best.data.metrics["f2"],
                "fraud_threshold": float(best.data.params.get("fraud_threshold", .5))}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
