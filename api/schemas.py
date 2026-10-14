from datetime import datetime
from typing import Annotated
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_timestamp: datetime
    customer_age: Annotated[int, Field(ge=18, le=100)]
    country: str
    membership_tier: str
    customer_tenure_days: Annotated[int, Field(ge=0)]
    product_category: str
    items_count: Annotated[int, Field(ge=1)]
    unit_price: Annotated[float, Field(gt=0)]
    discount_percent: Annotated[float, Field(ge=0, le=100)]
    order_value: Annotated[float, Field(gt=0)]
    payment_method: str
    device_type: str
    shipping_distance_km: Annotated[float, Field(ge=0)]
    delivery_days_estimate: Annotated[int, Field(ge=1)]
    is_weekend: Annotated[int, Field(ge=0, le=1)]
    hour_of_day: Annotated[int, Field(ge=0, le=23)]
    previous_orders: Annotated[int, Field(ge=0)]
    previous_chargebacks: Annotated[int, Field(ge=0)]


class MonitorRequest(BaseModel):
    transactions: list[TransactionRequest]
    serving_columns: list[str] | None = None
    preprocessing_version: str = "1.0.0"


class DriftRequest(BaseModel):
    """A minimum batch size prevents misleading one-record drift results."""
    transactions: list[TransactionRequest] = Field(min_length=30, max_length=1000)


class SkewRequest(BaseModel):
    serving_columns: list[str]
    preprocessing_version: str = "1.0.0"


class HealthRequest(BaseModel):
    validation_passed: bool
    drift_detected: bool
    drifted_feature_count: Annotated[int, Field(ge=0)] = 0
    anomaly_detected: bool
    skew_detected: bool
