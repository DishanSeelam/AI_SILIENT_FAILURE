"""Training-serving skew checks for feature and preprocessing contracts."""


def assess_skew(training_profile: dict, serving_profile: dict) -> dict:
    """Compare expected and serving feature contracts and report mismatches."""
    expected_columns = training_profile["feature_columns"]
    received_columns = serving_profile["feature_columns"]
    missing_features = [column for column in expected_columns if column not in received_columns]
    unexpected_features = [column for column in received_columns if column not in expected_columns]
    order_matches = expected_columns == received_columns
    expected_version = training_profile["preprocessing_version"]
    received_version = serving_profile["preprocessing_version"]
    issues: list[str] = []
    if missing_features:
        issues.append(f"Missing features: {', '.join(missing_features)}.")
    if unexpected_features:
        issues.append(f"Unexpected features: {', '.join(unexpected_features)}.")
    if not order_matches and not missing_features and not unexpected_features:
        issues.append("Feature order differs from the training contract.")
    if expected_version != received_version:
        issues.append(
            f"Preprocessing version mismatch: expected '{expected_version}', "
            f"received '{received_version}'."
        )
    return {
        "status": "skew_detected" if issues else "consistent",
        "missing_features": missing_features,
        "unexpected_features": unexpected_features,
        "feature_order_matches": order_matches,
        "preprocessing_version_matches": expected_version == received_version,
        "issues": issues,
    }
