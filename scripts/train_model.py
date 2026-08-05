"""Train and evaluate the baseline e-commerce fraud classifier.

Run from the project root using the project's virtual-environment Python.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utils.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, prepare_features
from monitoring.anomaly import ANOMALY_FEATURES, build_anomaly_input


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
REFERENCE_PROFILE_PATH = PROJECT_ROOT / "models" / "reference_profile.json"
REFERENCE_DATA_PATH = PROJECT_ROOT / "models" / "reference_features.csv"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.joblib"
TRAINING_CONTRACT_PATH = PROJECT_ROOT / "models" / "training_contract.json"
TARGET_COLUMN = "is_fraudulent"
RANDOM_STATE = 42

def build_pipeline() -> Pipeline:
    """Build one pipeline so training and inference use identical preprocessing."""
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def create_reference_profile(features: pd.DataFrame) -> dict[str, object]:
    """Save baseline statistics for later drift and skew detection."""
    numeric = {
        column: {
            "mean": round(float(features[column].mean()), 6),
            "std": round(float(features[column].std()), 6),
            "min": round(float(features[column].min()), 6),
            "max": round(float(features[column].max()), 6),
            "missing_rate": round(float(features[column].isna().mean()), 6),
        }
        for column in NUMERIC_FEATURES
    }
    categorical = {
        column: {
            "categories": sorted(features[column].dropna().astype(str).unique().tolist()),
            "missing_rate": round(float(features[column].isna().mean()), 6),
        }
        for column in CATEGORICAL_FEATURES
    }
    return {"feature_columns": FEATURE_COLUMNS, "numeric_features": numeric, "categorical_features": categorical}


def select_alert_threshold(labels: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    """Choose a threshold that favours catching fraud over raw accuracy.

    F2 weights recall twice as strongly as precision.  This is appropriate for a
    first-pass fraud alert: suspicious transactions can be reviewed manually.
    """
    best_threshold = 0.50
    best_result = {"precision": 0.0, "recall": 0.0, "f2": -1.0}
    for threshold in np.arange(0.05, 0.51, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        f2 = (5 * precision * recall / (4 * precision + recall)) if precision + recall else 0.0
        if f2 > best_result["f2"]:
            best_threshold = float(round(threshold, 2))
            best_result = {"precision": float(precision), "recall": float(recall), "f2": float(f2)}
    return best_threshold, best_result


def evaluate_predictions(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    """Return threshold-dependent and threshold-independent evaluation metrics."""
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "threshold": threshold,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    features = prepare_features(data)
    target = data[TARGET_COLUMN].astype(int)
    development_x, test_x, development_y, test_y = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=RANDOM_STATE
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        development_x,
        development_y,
        test_size=0.25,
        stratify=development_y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(train_x, train_y)
    validation_probabilities = pipeline.predict_proba(validation_x)[:, 1]
    alert_threshold, validation_selection = select_alert_threshold(validation_y, validation_probabilities)

    probabilities = pipeline.predict_proba(test_x)[:, 1]
    metrics = {
        "dataset_rows": int(len(data)),
        "training_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)),
        "test_rows": int(len(test_x)),
        "target": TARGET_COLUMN,
        "fraud_rate": round(float(target.mean()), 6),
        "roc_auc": round(float(roc_auc_score(test_y, probabilities)), 6),
        "average_precision": round(float(average_precision_score(test_y, probabilities)), 6),
        "validation_threshold_selection": {
            "method": "maximum_f2_score",
            "selected_threshold": alert_threshold,
            "validation_metrics": {key: round(value, 6) for key, value in validation_selection.items()},
        },
        "test_metrics_at_default_threshold": evaluate_predictions(test_y, probabilities, 0.50),
        "test_metrics_at_selected_threshold": evaluate_predictions(test_y, probabilities, alert_threshold),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Refit on all non-test data for the production artifact after evaluation.
    production_pipeline = build_pipeline()
    production_pipeline.fit(development_x, development_y)
    development_probabilities = production_pipeline.predict_proba(development_x)[:, 1]
    anomaly_training_data = pd.concat(
        [
            build_anomaly_input(row.to_dict(), probability)
            for (_, row), probability in zip(development_x.iterrows(), development_probabilities)
        ],
        ignore_index=True,
    )
    anomaly_detector = IsolationForest(
        contamination=0.03, random_state=RANDOM_STATE, n_estimators=200
    )
    anomaly_detector.fit(anomaly_training_data[ANOMALY_FEATURES])
    joblib.dump(
        {
            "pipeline": production_pipeline,
            "alert_threshold": alert_threshold,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )
    joblib.dump(anomaly_detector, ANOMALY_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REFERENCE_PROFILE_PATH.write_text(
        json.dumps(create_reference_profile(development_x), indent=2), encoding="utf-8"
    )
    development_x.to_csv(REFERENCE_DATA_PATH, index=False)
    TRAINING_CONTRACT_PATH.write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "preprocessing_version": "v1",
                "description": "Timestamp-derived month/day plus median imputation and one-hot encoding.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_predictions = (probabilities >= alert_threshold).astype(int)
    print("Fraud model training and threshold selection complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved anomaly detector: {ANOMALY_MODEL_PATH}")
    print(f"Saved training-serving contract: {TRAINING_CONTRACT_PATH}")
    print(f"Saved drift reference data: {REFERENCE_DATA_PATH}")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(classification_report(test_y, selected_predictions, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
