"""Numerical data-drift detection using PSI and Kolmogorov-Smirnov tests."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config.settings import settings
from utils.features import NUMERIC_FEATURES


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI; values above 0.20 indicate meaningful distribution change."""
    reference_values = reference.dropna().astype(float).to_numpy()
    current_values = current.dropna().astype(float).to_numpy()
    if len(reference_values) == 0 or len(current_values) == 0:
        return 0.0

    edges = np.unique(np.percentile(reference_values, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_count, _ = np.histogram(reference_values, bins=edges)
    current_count, _ = np.histogram(current_values, bins=edges)
    epsilon = 1e-6
    reference_share = reference_count / len(reference_values) + epsilon
    current_share = current_count / len(current_values) + epsilon
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def assess_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """Assess numerical feature drift for a current batch against a baseline."""
    feature_results: dict[str, dict[str, float | bool]] = {}
    drifted_features: list[str] = []
    for column in NUMERIC_FEATURES:
        psi = population_stability_index(reference_data[column], current_data[column])
        ks_statistic, ks_p_value = ks_2samp(
            reference_data[column].dropna(), current_data[column].dropna()
        )
        has_drift = psi >= settings.psi_warning_threshold or ks_p_value < settings.ks_significance_level
        feature_results[column] = {
            "psi": round(float(psi), 4),
            "ks_statistic": round(float(ks_statistic), 4),
            "ks_p_value": round(float(ks_p_value), 6),
            "drift_detected": bool(has_drift),
        }
        if has_drift:
            drifted_features.append(column)

    return {
        "status": "drift_detected" if drifted_features else "stable",
        "batch_size": int(len(current_data)),
        "drifted_features": drifted_features,
        "feature_results": feature_results,
    }
