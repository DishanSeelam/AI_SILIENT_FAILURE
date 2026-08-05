"""SHAP-based local explanations for fraud alerts and anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _positive_class_values(shap_values: object) -> np.ndarray:
    """Normalize SHAP outputs from supported SHAP versions to one 1-D vector."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[0, :, 1]
    return values[0]


def explain_prediction(model_pipeline: object, features: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Return influential transformed features for a positive fraud probability."""
    import shap

    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    shap_values = shap.TreeExplainer(classifier).shap_values(transformed)
    contributions = _positive_class_values(shap_values)
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {
            "feature": str(feature_names[index]),
            "shap_value": round(float(contributions[index]), 5),
            "impact": "increases fraud risk" if contributions[index] > 0 else "decreases fraud risk",
        }
        for index in top_indices
    ]
