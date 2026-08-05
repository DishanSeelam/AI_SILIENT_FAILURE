"""Central, dependency-free application settings."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Configuration values to be expanded as monitoring modules are implemented."""

    app_name: str = "AI Silent Failure Detection"
    app_version: str = "0.1.0"
    model_path: Path = PROJECT_ROOT / "models" / "fraud_model.joblib"
    reference_profile_path: Path = PROJECT_ROOT / "models" / "reference_profile.json"
    reference_data_path: Path = PROJECT_ROOT / "models" / "reference_features.csv"
    anomaly_model_path: Path = PROJECT_ROOT / "models" / "anomaly_detector.joblib"
    training_contract_path: Path = PROJECT_ROOT / "models" / "training_contract.json"
    psi_warning_threshold: float = 0.20
    ks_significance_level: float = 0.05
    health_warning_threshold: float = 70.0
    health_critical_threshold: float = 40.0


settings = Settings()
