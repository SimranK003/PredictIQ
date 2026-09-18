"""Centralized application configuration.

All environment-specific values must be read from here, never hardcoded
or read ad-hoc via os.environ elsewhere in the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    environment: str = "development"

    # Database
    database_url: str = "postgresql+psycopg2://predictiq:predictiq@localhost:5432/predictiq"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "predictiq-churn"

    # API
    api_cors_origins: str = "http://localhost:3000"
    max_upload_size_mb: int = 25

    # Storage
    dataset_storage_dir: str = "storage/datasets"

    # Model promotion policy — see app/services/registry.py. These gate
    # POST /models/promote; they never trigger promotion automatically.
    model_promotion_required_metrics: str = "precision,recall,f1,roc_auc"
    model_promotion_require_artifact_check: bool = True

    # Logging
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def model_promotion_required_metrics_list(self) -> list[str]:
        return [m.strip() for m in self.model_promotion_required_metrics.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
