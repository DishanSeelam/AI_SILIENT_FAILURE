"""Central paths and runtime settings."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "ecommerce_transactions.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
REFERENCE_PATH = MODELS_DIR / "reference_features.csv"
PROFILE_PATH = MODELS_DIR / "reference_profile.json"
CONTRACT_PATH = MODELS_DIR / "training_contract.json"
ANOMALY_PATH = MODELS_DIR / "anomaly_detector.joblib"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
MLFLOW_DB = ROOT_DIR / "database" / "mlflow_tracking.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "Fraud Detection Experiments"
REGISTERED_MODEL_NAME = "FraudDetectionModel"

NUMERIC_FEATURES = [
    "customer_age", "customer_tenure_days", "items_count", "unit_price",
    "discount_percent", "order_value", "shipping_distance_km",
    "delivery_days_estimate", "is_weekend", "hour_of_day",
    "previous_orders", "previous_chargebacks",
]
CATEGORICAL_FEATURES = [
    "country", "membership_tier", "product_category", "payment_method", "device_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
