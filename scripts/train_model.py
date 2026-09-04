"""Train and evaluate the baseline e-commerce fraud classifier.

Run from the project root using the project's virtual-environment Python.
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    fbeta_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utils.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, prepare_features
from monitoring.anomaly import ANOMALY_FEATURES, build_anomaly_input
from utils.mlflow_tracking import configure_mlflow, write_model_manifest
from config.settings import settings


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
REFERENCE_PROFILE_PATH = PROJECT_ROOT / "models" / "reference_profile.json"
REFERENCE_DATA_PATH = PROJECT_ROOT / "models" / "reference_features.csv"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.joblib"
TRAINING_CONTRACT_PATH = PROJECT_ROOT / "models" / "training_contract.json"
TARGET_COLUMN = "is_fraudulent"
RANDOM_STATE = 42

def build_pipeline(model_type: str, n_estimators: int, min_samples_leaf: int) -> Pipeline:
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
    if model_type == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    else:
        classifier = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
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
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "f2": round(float(fbeta_score(labels, predictions, beta=2, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def parse_args() -> argparse.Namespace:
    """Support repeatable experiment runs without editing source code."""
    parser = argparse.ArgumentParser(description="Train and track a fraud-detection experiment.")
    parser.add_argument("--model-type", choices=["random_forest", "logistic_regression"], default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--run-name", default="fraud-training")
    parser.add_argument("--register", action="store_true", help="Register this run as a new FraudDetectionModel version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    pipeline = build_pipeline(args.model_type, args.n_estimators, args.min_samples_leaf)
    pipeline.fit(train_x, train_y)
    validation_probabilities = pipeline.predict_proba(validation_x)[:, 1]
    alert_threshold, validation_selection = select_alert_threshold(validation_y, validation_probabilities)

    probabilities = pipeline.predict_proba(test_x)[:, 1]
    metrics = {
        "dataset_rows": int(len(data)),
        "training_rows": int(len(train_x)),
        "validation_rows": int(len(validation_x)),
        "test_rows": int(len(test_x)),
        "target": TARGET_COLUMN, "model_type": args.model_type,
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
    production_pipeline = build_pipeline(args.model_type, args.n_estimators, args.min_samples_leaf)
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

    mlflow = configure_mlflow()
    parameters = {
        "model_type": args.model_type, "random_state": RANDOM_STATE, "test_size": 0.20,
        "validation_size": 0.25, "fraud_threshold": alert_threshold,
        "n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
    }
    tracked_metrics = {
        "accuracy": metrics["test_metrics_at_selected_threshold"]["accuracy"],
        "precision": metrics["test_metrics_at_selected_threshold"]["precision"],
        "recall": metrics["test_metrics_at_selected_threshold"]["recall"],
        "f1": metrics["test_metrics_at_selected_threshold"]["f1"],
        "f2": metrics["test_metrics_at_selected_threshold"]["f2"],
        "roc_auc": metrics["roc_auc"], "pr_auc": metrics["average_precision"],
    }
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(tracked_metrics)
        manifest = {
            "model_version": "local", "model_stage": "Fallback", "run_id": run.info.run_id,
            "model_uri": "", "alert_threshold": alert_threshold,
        }
        try:
            mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
            mlflow.log_artifact(str(REFERENCE_PROFILE_PATH), artifact_path="monitoring")
            mlflow.log_artifact(str(TRAINING_CONTRACT_PATH), artifact_path="monitoring")
            # This is a local, project-owned sklearn pipeline. Pickle avoids MLflow 3's
            # skops trusted-type restriction for numpy.dtype within sklearn artifacts.
            mlflow.sklearn.log_model(
                production_pipeline,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            run_model_uri = f"runs:/{run.info.run_id}/model"
            manifest["model_uri"] = run_model_uri
            if args.register:
                registered = mlflow.register_model(run_model_uri, settings.registered_model_name)
                client = mlflow.tracking.MlflowClient()
                client.set_model_version_tag(settings.registered_model_name, registered.version, "alert_threshold", str(alert_threshold))
                client.set_model_version_tag(settings.registered_model_name, registered.version, "model_stage", "Production")
                manifest.update({
                    "model_version": str(registered.version), "model_stage": "Production",
                    "model_uri": f"models:/{settings.registered_model_name}/{registered.version}",
                })
        except Exception as error:
            # Metrics remain valid; preserve diagnostics while allowing the run to finish.
            warning = f"Artifact or registry logging warning: {type(error).__name__}: {error}"
            mlflow.set_tag("tracking_warning", warning[:4500])
            (PROJECT_ROOT / "logs" / "mlflow_tracking_errors.log").open("a", encoding="utf-8").write(
                f"{args.run_name}: {warning}\n"
            )
        write_model_manifest(manifest)

    selected_predictions = (probabilities >= alert_threshold).astype(int)
    print("Fraud model training and threshold selection complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved anomaly detector: {ANOMALY_MODEL_PATH}")
    print(f"Saved training-serving contract: {TRAINING_CONTRACT_PATH}")
    print(f"Saved drift reference data: {REFERENCE_DATA_PATH}")
    print(f"MLflow run logged for experiment: {settings.mlflow_experiment_name}")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(classification_report(test_y, selected_predictions, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
