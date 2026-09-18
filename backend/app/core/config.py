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

    # Inference / batch prediction
    batch_prediction_sync_max_records: int = 50
    batch_prediction_max_records: int = 5000

    # Drift detection (see monitoring/drift.py and docs/monitoring.md for
    # full methodology). These are *operational* thresholds tuned for
    # this project's scale and traffic, not universal statistical
    # constants — they gate an alert-worthy signal, not a proof of drift.
    #
    # 100 is the conventional "reasonably safe" floor for a two-sample KS
    # test to have workable power against a moderate shift, and it also
    # gives PSI's categorical bins (some Telco categories are rare, e.g.
    # a specific payment method) enough counts each to not be pure noise.
    # Below it, we report insufficient_data rather than a number that
    # would fluctuate wildly between requests based on a handful of rows.
    drift_min_samples: int = 100
    # KS statistic threshold (numeric features). 0.10 is a commonly used
    # starting point in industry drift monitoring — not derived from this
    # project's data — meaning "the two empirical CDFs differ by at least
    # 10 percentage points at their point of maximum divergence."
    drift_ks_threshold: float = 0.10
    # PSI thresholds (categorical features), the standard banding used in
    # credit-risk/ML-ops literature: <0.10 no significant change, 0.10-
    # 0.25 moderate ("warning"), >0.25 substantial ("critical"). These are
    # conventions, not laws of statistics.
    drift_psi_warning: float = 0.10
    drift_psi_critical: float = 0.25
    # Safety cap on how many recent predictions are pulled into memory
    # for a single drift computation, regardless of how large the
    # matched sample actually is.
    drift_max_predictions_sampled: int = 5000

    # Async training jobs — retries apply only to transient infra errors
    # (DB/MLflow connectivity), never to deterministic data/training
    # failures. See app/services/training.py.
    training_job_max_retries: int = 2
    training_job_retry_delay_seconds: int = 15

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
