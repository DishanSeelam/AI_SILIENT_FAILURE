"""Feature validation checks before model inference."""

import json
from functools import lru_cache
from pathlib import Path

from config.settings import settings


@lru_cache(maxsize=1)
def load_reference_profile() -> dict:
    """Load categories and feature ranges recorded during model training."""
    profile_path = Path(settings.reference_profile_path)
    if not profile_path.exists():
        raise FileNotFoundError("Reference profile not found. Train the model first.")
    return json.loads(profile_path.read_text(encoding="utf-8"))


def validate_features(features: dict) -> list[str]:
    """Return human-readable schema/profile errors for an inference payload.

    Basic types and absolute ranges are enforced by Pydantic. This check uses the
    saved training profile to catch category and range changes that can silently
    break a running model.
    """
    profile = load_reference_profile()
    errors: list[str] = []

    for column, details in profile["categorical_features"].items():
        value = str(features.get(column, ""))
        allowed_categories = details["categories"]
        if value not in allowed_categories:
            errors.append(
                f"{column} has unexpected value '{value}'. "
                f"Allowed values: {', '.join(allowed_categories)}."
            )

    # Use broad tolerance around the training range: unusual valid values are
    # logged later by drift monitoring, whereas impossible values are rejected.
    for column, details in profile["numeric_features"].items():
        if column not in features:
            continue
        value = float(features[column])
        lower_bound = float(details["min"]) - max(1.0, abs(float(details["min"])) * 0.5)
        upper_bound = float(details["max"]) + max(1.0, abs(float(details["max"])) * 0.5)
        if value < lower_bound or value > upper_bound:
            errors.append(
                f"{column}={value} is outside the accepted range "
                f"({lower_bound:.2f} to {upper_bound:.2f})."
            )
    return errors
