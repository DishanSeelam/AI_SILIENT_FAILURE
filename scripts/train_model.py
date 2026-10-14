"""Train and evaluate one candidate fraud model, logging a reproducible MLflow run."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import (ANOMALY_PATH, CONTRACT_PATH, DATA_PATH, FEATURE_COLUMNS,
                             METRICS_PATH, MODELS_DIR, PROFILE_PATH, REFERENCE_PATH,
                             MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
from utils.features import build_preprocessor


def make_estimator(model_type: str, n_estimators: int, min_samples_leaf: int, seed: int,
                   scale_pos_weight: float, max_depth: int, learning_rate: float):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)
    common = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                  class_weight="balanced", random_state=seed, n_jobs=-1)
    if model_type == "random_forest":
        return RandomForestClassifier(**common)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(**common)
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("XGBoost is not installed. Run: pip install xgboost") from error
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=.85, colsample_bytree=.85, min_child_weight=min_samples_leaf,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=seed,
            n_jobs=-1, tree_method="hist",
        )
    raise ValueError(f"Unknown model type: {model_type}")


def select_threshold(y_true, probabilities):
    candidates = np.arange(.10, .61, .01)
    scores = [fbeta_score(y_true, probabilities >= t, beta=2, zero_division=0) for t in candidates]
    index = int(np.argmax(scores))
    return float(candidates[index]), float(scores[index])


def main(args):
    data = pd.read_csv(DATA_PATH)
    X, y = data[FEATURE_COLUMNS], data["is_fraudulent"]
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_development, y_development, test_size=args.validation_size,
        random_state=args.random_state, stratify=y_development)
    imbalance_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pipeline = Pipeline([("preprocessor", build_preprocessor()),
                         ("classifier", make_estimator(args.model_type, args.n_estimators,
                                                        args.min_samples_leaf, args.random_state,
                                                        imbalance_ratio, args.max_depth, args.learning_rate))])
    pipeline.fit(X_train, y_train)
    threshold, validation_f2 = select_threshold(y_valid, pipeline.predict_proba(X_valid)[:, 1])
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "f2": fbeta_score(y_test, prediction, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "validation_f2": validation_f2,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{args.model_type}_candidate.joblib")
    # A local serving fallback is intentionally retained; best-model selection overwrites it.
    joblib.dump(pipeline, MODELS_DIR / "fraud_model.joblib")
    X_train.to_csv(REFERENCE_PATH, index=False)
    profile = {"numeric": {column: {"min": float(X_train[column].min()), "max": float(X_train[column].max()),
                              "mean": float(X_train[column].mean()), "std": float(X_train[column].std() or 0)}
                           for column in X_train.select_dtypes(include="number").columns},
               "categorical": {column: sorted(X_train[column].astype(str).unique().tolist())
                               for column in X_train.select_dtypes(exclude="number").columns}}
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    CONTRACT_PATH.write_text(json.dumps({"feature_columns": FEATURE_COLUMNS,
                                         "preprocessing_version": "1.0.0"}, indent=2), encoding="utf-8")
    anomaly_features = ["fraud_probability", "order_value", "customer_tenure_days", "hour_of_day",
                        "previous_chargebacks", "shipping_distance_km"]
    anomaly_train = X_train[[x for x in anomaly_features if x != "fraud_probability"]].copy()
    anomaly_train.insert(0, "fraud_probability", pipeline.predict_proba(X_train)[:, 1])
    anomaly_model = IsolationForest(contamination=.03, random_state=args.random_state)
    anomaly_model.fit(anomaly_train[anomaly_features])
    joblib.dump({"model": anomaly_model, "features": anomaly_features}, ANOMALY_PATH)
    result = {"model_type": args.model_type, "threshold": threshold, "rows": len(data),
              "fraud_rate": float(y.mean()), "metrics": {key: round(float(value), 6) for key, value in metrics.items()}}
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    params = {"model_type": args.model_type, "random_state": args.random_state,
              "test_size": args.test_size, "validation_size": args.validation_size,
              "fraud_threshold": threshold, "n_estimators": args.n_estimators,
              "min_samples_leaf": args.min_samples_leaf, "max_depth": args.max_depth,
              "learning_rate": args.learning_rate, "scale_pos_weight": imbalance_ratio,
              "dataset_rows": len(data)}
    with mlflow.start_run(run_name=args.run_name or args.model_type) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tags({"dataset_type": "synthetic", "selection_metric": "f2",
                         "model_role": "candidate"})
        mlflow.log_artifact(str(METRICS_PATH), "evaluation")
        mlflow.log_artifact(str(PROFILE_PATH), "monitoring")
        mlflow.log_artifact(str(CONTRACT_PATH), "monitoring")
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")
        result["run_id"] = run.info.run_id
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["logistic_regression", "random_forest", "extra_trees", "xgboost"], default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=.20)
    parser.add_argument("--validation-size", type=float, default=.25)
    parser.add_argument("--run-name", default=None)
    main(parser.parse_args())
