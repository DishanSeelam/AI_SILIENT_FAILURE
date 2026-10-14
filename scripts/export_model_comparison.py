"""Export an auditable comparison table from completed MLflow runs."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def main():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise RuntimeError("Experiment not found. Train candidates first.")
    records = []
    for run in client.search_runs([experiment.experiment_id], max_results=100):
        if "f2" not in run.data.metrics:
            continue
        records.append({"run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                        "model_type": run.data.params.get("model_type", "unknown"),
                        **{name: round(run.data.metrics.get(name, float("nan")), 4)
                           for name in ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]}})
    table = pd.DataFrame(records).sort_values(["f2", "pr_auc"], ascending=False)
    table.to_csv(ROOT / "logs" / "model_comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
