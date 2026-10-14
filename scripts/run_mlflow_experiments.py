"""Run comparable baseline and ensemble fraud-model experiments."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("logistic-baseline", ["--model-type", "logistic_regression"]),
    ("rf-baseline", ["--model-type", "random_forest", "--n-estimators", "300", "--min-samples-leaf", "3"]),
    ("extra-trees", ["--model-type", "extra_trees", "--n-estimators", "400", "--min-samples-leaf", "2"]),
    ("rf-optimized", ["--model-type", "random_forest", "--n-estimators", "600", "--min-samples-leaf", "1"]),
    ("xgboost-baseline", ["--model-type", "xgboost", "--n-estimators", "350", "--max-depth", "5", "--learning-rate", "0.05", "--min-samples-leaf", "2"]),
    ("xgboost-optimized", ["--model-type", "xgboost", "--n-estimators", "550", "--max-depth", "4", "--learning-rate", "0.04", "--min-samples-leaf", "2"]),
]

for name, options in RUNS:
    print(f"\n--- {name} ---")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_model.py"), "--run-name", name, *options],
                   check=True, cwd=ROOT)
print("\nCompleted six MLflow experiments. Select the top F2 run before deployment.")
