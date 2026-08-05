"""Demonstrate data-drift detection with a deliberately shifted production batch.

Run from the project root using the project's virtual-environment Python.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import settings
from monitoring.drift import assess_drift


RESULT_PATH = PROJECT_ROOT / "logs" / "drift_experiment_result.json"
RANDOM_STATE = 42


def main() -> None:
    """Build a shifted batch and report the detector's real result."""
    if not settings.reference_data_path.exists():
        raise FileNotFoundError("Reference data is missing. Run scripts/train_model.py first.")

    reference_data = pd.read_csv(settings.reference_data_path)
    current_batch = reference_data.sample(n=200, replace=False, random_state=RANDOM_STATE).copy()

    # Intentional production change: a new high-value, long-distance customer segment.
    current_batch["order_value"] = current_batch["order_value"] * 3.5
    current_batch["unit_price"] = current_batch["unit_price"] * 2.8
    current_batch["customer_age"] = np.minimum(current_batch["customer_age"] + 25, 100)
    current_batch["shipping_distance_km"] = current_batch["shipping_distance_km"] + 900

    result = assess_drift(reference_data, current_batch)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Drift experiment complete.")
    print("Injected shifts: order value ×3.5, unit price ×2.8, age +25, distance +900 km.")
    print(json.dumps(result, indent=2))
    print(f"Saved full result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
