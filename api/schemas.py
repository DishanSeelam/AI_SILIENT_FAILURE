"""Request and response schemas for the prediction service."""

from datetime import datetime

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """A single e-commerce transaction submitted for fraud assessment."""

    transaction_timestamp: datetime
    customer_age: int = Field(ge=18, le=100)
    country: str = Field(min_length=2, max_length=60)
    membership_tier: str = Field(min_length=2, max_length=30)
    customer_tenure_days: int = Field(ge=0)
    product_category: str = Field(min_length=2, max_length=60)
    items_count: int = Field(ge=1, le=100)
    unit_price: float = Field(gt=0)
    discount_percent: float = Field(ge=0, le=100)
    order_value: float = Field(gt=0)
    payment_method: str = Field(min_length=2, max_length=40)
    device_type: str = Field(min_length=2, max_length=30)
    shipping_distance_km: float = Field(ge=0)
    delivery_days_estimate: int = Field(ge=1, le=60)
    is_weekend: int = Field(ge=0, le=1)
    hour_of_day: int = Field(ge=0, le=23)
    previous_orders: int = Field(ge=0)
    previous_chargebacks: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Model inference result returned to API clients."""

    fraud_probability: float
    alert_threshold: float
    predicted_label: int
    decision: str
    confidence: float
    validation_status: str
    anomaly_detected: bool
    anomaly_score: float
    explanation_generated: bool
    top_feature_contributions: list[dict] = []


class DriftRequest(BaseModel):
    """A batch of recent transactions to compare with the training baseline."""

    transactions: list[TransactionRequest] = Field(min_length=30, max_length=2_000)


class SkewRequest(BaseModel):
    """Declared feature contract of the serving pipeline to validate."""

    feature_columns: list[str] = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1, max_length=40)


class HealthRequest(BaseModel):
    """Current monitoring signals supplied to the unified health engine."""

    validation_passed: bool = True
    drift_detected: bool = False
    drifted_feature_count: int = Field(default=0, ge=0)
    anomaly_detected: bool = False
    skew_detected: bool = False
