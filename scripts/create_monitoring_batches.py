"""Create valid normal and intentionally drifted API batch payloads."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def serialise(frame):
    result = frame.drop(columns=["is_fraudulent"], errors="ignore").copy()
    result["transaction_timestamp"] = pd.to_datetime(result["transaction_timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"transactions": result.to_dict("records")}


def main():
    logs = ROOT / "logs"; logs.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "raw" / "ecommerce_transactions.csv")
    normal = data.sample(50, random_state=42)
    drift = normal.copy()
    # Intentionally shifted while remaining valid under the training feature contract.
    drift["customer_age"] = 70; drift["customer_tenure_days"] = 3
    drift["unit_price"] = 1100; drift["order_value"] = 2600
    drift["shipping_distance_km"] = 1300; drift["hour_of_day"] = 2
    drift["payment_method"] = "Credit Card"; drift["device_type"] = "Desktop"
    (logs / "normal_batch.json").write_text(json.dumps(serialise(normal), indent=2), encoding="utf-8")
    (logs / "drift_batch.json").write_text(json.dumps(serialise(drift), indent=2), encoding="utf-8")
    print("Created logs/normal_batch.json and logs/drift_batch.json (50 records each).")


if __name__ == "__main__":
    main()
