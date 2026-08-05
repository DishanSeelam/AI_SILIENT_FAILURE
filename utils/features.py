"""Shared feature engineering used identically in training and serving."""

import pandas as pd


NUMERIC_FEATURES = [
    "customer_age", "customer_tenure_days", "items_count", "unit_price",
    "discount_percent", "order_value", "shipping_distance_km",
    "delivery_days_estimate", "is_weekend", "hour_of_day", "previous_orders",
    "previous_chargebacks", "purchase_month", "purchase_day_of_week",
]
CATEGORICAL_FEATURES = [
    "country", "membership_tier", "product_category", "payment_method", "device_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw transaction fields."""
    prepared = data.copy()
    timestamp = pd.to_datetime(prepared["transaction_timestamp"], errors="raise")
    prepared["purchase_month"] = timestamp.dt.month
    prepared["purchase_day_of_week"] = timestamp.dt.dayofweek
    return prepared[FEATURE_COLUMNS]
