"""Generate a reproducible synthetic e-commerce fraud dataset.

The data contains realistic-looking relationships but is entirely simulated.  It is
for system evaluation only, not a substitute for a real, governed fraud dataset.
"""
import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ecommerce_transactions.csv"
FIELDS = [
    "transaction_timestamp", "customer_age", "country", "membership_tier",
    "customer_tenure_days", "product_category", "items_count", "unit_price",
    "discount_percent", "order_value", "payment_method", "device_type",
    "shipping_distance_km", "delivery_days_estimate", "is_weekend", "hour_of_day",
    "previous_orders", "previous_chargebacks", "is_fraudulent",
]


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-25, min(25, x))))


def make_row(rng: random.Random, start: datetime) -> dict:
    timestamp = start + timedelta(minutes=rng.randrange(365 * 24 * 60))
    country = rng.choices(["India", "United States", "United Kingdom", "Germany", "UAE"], [50, 17, 12, 11, 10])[0]
    tier = rng.choices(["Basic", "Standard", "Premium"], [0.40, 0.45, 0.15])[0]
    category = rng.choices(["Electronics", "Fashion", "Groceries", "Beauty", "Home"], [0.28, .25, .19, .12, .16])[0]
    payment = rng.choices(["Credit Card", "Debit Card", "UPI", "Wallet", "Cash on Delivery"], [.34, .21, .24, .13, .08])[0]
    device = rng.choices(["Mobile", "Desktop", "Tablet"], [.69, .25, .06])[0]
    age = max(18, min(80, round(rng.gauss(34, 11))))
    tenure = max(0, int(rng.expovariate(1 / 520)))
    items = rng.randint(1, 7)
    unit_price = round(max(40, rng.lognormvariate(6.2, .72)), 2)
    discount = rng.choices([0, 5, 10, 15, 20, 30, 40], [30, 18, 20, 14, 9, 6, 3])[0]
    order_value = round(items * unit_price * (1 - discount / 100), 2)
    distance = round(min(2500, rng.lognormvariate(5.8, .82)), 1)
    delivery = max(1, min(12, round(2 + distance / 310 + rng.gauss(0, 1.2))))
    previous_orders = max(0, int(rng.expovariate(1 / 18)))
    chargebacks = rng.choices([0, 1, 2, 3], [.94, .045, .012, .003])[0]
    weekend = int(timestamp.weekday() >= 5)
    hour = timestamp.hour
    # Signal is deliberately distributed across several features: no single target-leaking column.
    risk = -3.15
    risk += 1.45 * min(chargebacks, 2)
    risk += 1.00 * int(tenure < 25) + .55 * int(previous_orders == 0)
    risk += .85 * int(order_value > 2400) + .48 * int(distance > 1200)
    risk += .48 * int(hour <= 4) + .32 * int(payment == "Credit Card")
    risk += .30 * int(device == "Desktop") + .25 * int(category == "Electronics")
    risk += .25 * int(discount >= 30) + rng.gauss(0, .72)
    fraud = int(rng.random() < sigmoid(risk))
    return {
        "transaction_timestamp": timestamp.isoformat(), "customer_age": age, "country": country,
        "membership_tier": tier, "customer_tenure_days": tenure, "product_category": category,
        "items_count": items, "unit_price": unit_price, "discount_percent": discount,
        "order_value": order_value, "payment_method": payment, "device_type": device,
        "shipping_distance_km": distance, "delivery_days_estimate": delivery,
        "is_weekend": weekend, "hour_of_day": hour, "previous_orders": previous_orders,
        "previous_chargebacks": chargebacks, "is_fraudulent": fraud,
    }


def main(rows: int, seed: int) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for _ in range(rows):
            writer.writerow(make_row(rng, start))
    print(f"Created {OUT} with {rows:,} synthetic rows (seed={seed}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.rows, args.seed)
