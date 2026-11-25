"""Serving input validation against the training reference profile."""
import json
from config.settings import PROFILE_PATH


def validate_record(record: dict):
    if not PROFILE_PATH.exists():
        return {"status": "unavailable", "issues": ["Train a model first."]}
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    issues = []
    # Support earlier training artifacts while new runs use the compact numeric/categorical schema.
    numeric_profile = profile.get("numeric", profile.get("numeric_features", {}))
    categorical_profile = profile.get("categorical", profile.get("categorical_features", {}))
    for feature, bounds in numeric_profile.items():
        value = record.get(feature)
        if value is None:
            # Derived training-only fields (e.g., date features) are not expected in the API payload.
            continue
        lower = float(bounds["min"]) - max(1.0, abs(float(bounds["min"])) * .5)
        upper = float(bounds["max"]) + max(1.0, abs(float(bounds["max"])) * .5)
        if not (lower <= float(value) <= upper):
            issues.append(f"Out-of-reference range: {feature}")
    for feature, details in categorical_profile.items():
        allowed = details.get("categories", []) if isinstance(details, dict) else details
        if str(record.get(feature)) not in allowed:
            issues.append(f"Unexpected category for {feature}: {record.get(feature)}")
    return {"status": "passed" if not issues else "failed", "issues": issues}
