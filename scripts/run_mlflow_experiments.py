"""Run comparable tracked experiments and register the selected final model."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN_SCRIPT = ROOT / "scripts" / "train_model.py"

EXPERIMENTS = [
    ("logistic-baseline", ["--model-type", "logistic_regression"]),
    ("rf-baseline", ["--model-type", "random_forest", "--n-estimators", "200", "--min-samples-leaf", "3"]),
    ("rf-optimized", ["--model-type", "random_forest", "--n-estimators", "500", "--min-samples-leaf", "1", "--register"]),
]


def main() -> None:
    """Create three MLflow runs; the final optimized run becomes the active registry version."""
    for run_name, options in EXPERIMENTS:
        command = [PYTHON, str(TRAIN_SCRIPT), "--run-name", run_name, *options]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
    print("Completed three MLflow experiments. Open `mlflow ui --backend-store-uri ./mlruns` to compare them.")


if __name__ == "__main__":
    main()
