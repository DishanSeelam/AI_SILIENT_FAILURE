"""Generate a reproducible synthetic e-commerce fraud-detection dataset.

The data is deliberately synthetic: it contains no real customer information and
is suitable for a student MLOps project.  ``is_fraudulent`` is the target label.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


SEED = 20260718
ROW_COUNT = 2_000
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ecommerce_transactions.csv"

COUNTRIES = ["India", "United States", "United Kingdom", "Canada", "Australia", "Germany"]
CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Wallet", "Cash on Delivery"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
MEMBERSHIP_TIERS = ["Standard", "Silver", "Gold", "Platinum"]
FIELDNAMES = [
    "transaction_id",
    "transaction_timestamp",
    "customer_id",
    "customer_age",
    "country",
    "membership_tier",
    "customer_tenure_days",
    "product_category",
    "items_count",
    "unit_price",
    "discount_percent",
    "order_value",
    "payment_method",
    "device_type",
    "shipping_distance_km",
    "delivery_days_estimate",
    "is_weekend",
    "hour_of_day",
    "previous_orders",
    "previous_chargebacks",
    "is_fraudulent",
]


def generate_rows() -> list[dict[str, object]]:
    """Return realistic-but-synthetic transaction rows with a learnable target."""
    rng = random.Random(SEED)
    start = datetime(2025, 1, 1)
    rows: list[dict[str, object]] = []

    for number in range(1, ROW_COUNT + 1):
        timestamp = start + timedelta(minutes=rng.randint(0, 525_599))
        age = max(18, min(75, round(rng.gauss(34, 10))))
        category = rng.choice(CATEGORIES)
        items = rng.choices([1, 2, 3, 4, 5], weights=[45, 28, 15, 8, 4])[0]
        unit_price = round(rng.lognormvariate(4.2, 0.75), 2)
        discount = rng.choice([0, 0, 0, 5, 10, 15, 20, 30])
        order_value = round(items * unit_price * (1 - discount / 100), 2)
        payment = rng.choices(PAYMENT_METHODS, weights=[34, 23, 24, 12, 7])[0]
        device = rng.choices(DEVICES, weights=[65, 29, 6])[0]
        tenure = rng.randint(0, 2_500)
        previous_orders = max(0, int(rng.gauss(tenure / 100, 4)))
        chargebacks = 0 if rng.random() < 0.96 else rng.choice([1, 2])
        hour = timestamp.hour
        distance = round(rng.triangular(2, 1_500, 220), 1)
        delivery_days = max(1, min(12, round(2 + distance / 250 + rng.random() * 2)))

        # Higher-risk patterns create a non-random target for the future classifier.
        risk = 0.012
        risk += 0.21 if chargebacks > 0 else 0
        risk += 0.07 if order_value > 800 else 0
        risk += 0.05 if tenure < 30 else 0
        risk += 0.04 if hour in {0, 1, 2, 3, 4, 5} else 0
        risk += 0.03 if payment == "Credit Card" and device == "Desktop" else 0
        risk -= 0.02 if previous_orders > 15 else 0
        is_fraudulent = int(rng.random() < min(risk, 0.65))

        rows.append(
            {
                "transaction_id": f"TXN{number:07d}",
                "transaction_timestamp": timestamp.isoformat(timespec="minutes"),
                "customer_id": f"CUST{rng.randint(1, 850):05d}",
                "customer_age": age,
                "country": rng.choice(COUNTRIES),
                "membership_tier": rng.choices(MEMBERSHIP_TIERS, weights=[58, 24, 13, 5])[0],
                "customer_tenure_days": tenure,
                "product_category": category,
                "items_count": items,
                "unit_price": unit_price,
                "discount_percent": discount,
                "order_value": order_value,
                "payment_method": payment,
                "device_type": device,
                "shipping_distance_km": distance,
                "delivery_days_estimate": delivery_days,
                "is_weekend": int(timestamp.weekday() >= 5),
                "hour_of_day": hour,
                "previous_orders": previous_orders,
                "previous_chargebacks": chargebacks,
                "is_fraudulent": is_fraudulent,
            }
        )
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    fraud_count = sum(row["is_fraudulent"] for row in rows)
    print(f"Created {OUTPUT_PATH} with {len(rows)} rows ({fraud_count} fraudulent transactions).")


if __name__ == "__main__":
    main()
