"""Training configuration.

Hyperparameters and split settings are never hardcoded in the training
script. Defaults live here; they can be overridden by environment
variables (prefix TRAIN_, e.g. TRAIN_RANDOM_FOREST_N_ESTIMATORS=500) or
by a YAML file passed via --config, which takes precedence over env vars.
This is what "reproducible training" means in practice: the exact
resolved config is logged to MLflow with every run.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_", extra="ignore")

    # Split
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # Logistic Regression
    logistic_regression_c: float = 1.0
    logistic_regression_max_iter: int = 1000

    # Random Forest
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int | None = None
    random_forest_min_samples_leaf: int = 2

    # XGBoost
    xgboost_n_estimators: int = 300
    xgboost_max_depth: int = 5
    xgboost_learning_rate: float = 0.1
    xgboost_subsample: float = 0.9

    def resolved_dict(self) -> dict[str, Any]:
        return self.model_dump()


def load_training_config(config_path: str | Path | None = None) -> TrainingConfig:
    """Build the effective config: defaults <- env vars <- YAML file.

    A YAML file's values, if given, win over environment variables —
    it represents an explicit, reviewable choice for a specific run.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    overrides: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
    return TrainingConfig(**overrides)
