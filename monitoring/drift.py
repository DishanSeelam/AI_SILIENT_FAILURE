"""Batch distribution drift checks using PSI and Kolmogorov-Smirnov tests."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from config.settings import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref, _ = np.histogram(reference, bins=edges)
    cur, _ = np.histogram(current, bins=edges)
    ref_rate = np.clip(ref / max(ref.sum(), 1), 1e-6, None)
    cur_rate = np.clip(cur / max(cur.sum(), 1), 1e-6, None)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def categorical_psi(reference, current):
    cats = sorted(set(reference.astype(str)) | set(current.astype(str)))
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    total = 0.0
    for category in cats:
        a, b = max(ref.get(category, 0), 1e-6), max(cur.get(category, 0), 1e-6)
        total += (b - a) * np.log(b / a)
    return float(total)


def assess_drift(reference: pd.DataFrame, current: pd.DataFrame):
    results = []
    for column in NUMERIC_FEATURES:
        if column not in reference or column not in current:
            continue
        ref, cur = reference[column].dropna(), current[column].dropna()
        if len(ref) < 10 or len(cur) < 10:
            continue
        value = psi(ref, cur)
        _, p_value = ks_2samp(ref, cur)
        detected = value >= .20 or p_value < .05
        results.append({"feature": column, "type": "numeric", "psi": round(value, 4),
                        "ks_p_value": round(float(p_value), 6), "drift_detected": detected})
    for column in CATEGORICAL_FEATURES:
        if column not in reference or column not in current:
            continue
        value = categorical_psi(reference[column], current[column])
        unseen = sorted(set(current[column].astype(str)) - set(reference[column].astype(str)))
        detected = value >= .20 or bool(unseen)
        results.append({"feature": column, "type": "categorical", "psi": round(value, 4),
                        "unseen_categories": unseen, "drift_detected": detected})
    drifted = [item for item in results if item["drift_detected"]]
    return {"status": "drift_detected" if drifted else "stable", "drifted_features": len(drifted),
            "features": results, "thresholds": {"psi": .20, "ks_p_value": .05}}
