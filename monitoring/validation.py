"""Serving input validation against the training reference profile."""
import json
from config.settings import PROFILE_PATH


def validate_record(record: dict):
    if not PROFILE_PATH.exists():
        return {"status": "unavailable", "issues": ["Train a model first."]}
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    issues = []
    for feature, bounds in profile["numeric"].items():
        value = record.get(feature)
        if value is None:
            issues.append(f"Missing numeric feature: {feature}")
        elif not (bounds["min"] - 1e-9 <= float(value) <= bounds["max"] + 1e-9):
            issues.append(f"Out-of-reference range: {feature}")
    for feature, allowed in profile["categorical"].items():
        if str(record.get(feature)) not in allowed:
            issues.append(f"Unexpected category for {feature}: {record.get(feature)}")
    return {"status": "passed" if not issues else "failed", "issues": issues}
