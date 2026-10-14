"""Training-serving feature-contract checks."""
import json
from config.settings import CONTRACT_PATH


def assess_skew(serving_columns, preprocessing_version="1.0.0"):
    if not CONTRACT_PATH.exists():
        return {"status": "unavailable", "issues": ["Train a model first."]}
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected = contract["feature_columns"]
    missing = [item for item in expected if item not in serving_columns]
    unexpected = [item for item in serving_columns if item not in expected]
    order_matches = list(serving_columns) == expected
    version_matches = preprocessing_version == contract["preprocessing_version"]
    issues = ([f"Missing feature: {x}" for x in missing] + [f"Unexpected feature: {x}" for x in unexpected] +
              ([] if order_matches else ["Feature order differs from the training contract."]) +
              ([] if version_matches else ["Preprocessing version differs from the training contract."]))
    return {"status": "consistent" if not issues else "skew_detected", "missing_features": missing,
            "unexpected_features": unexpected, "feature_order_matches": order_matches,
            "preprocessing_version_matches": version_matches, "issues": issues}
