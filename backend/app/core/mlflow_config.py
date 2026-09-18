"""Centralized MLflow tracking configuration.

MLflow's tracking URI is process-global state, not shared across
processes — Phase 4 hit this bug twice (once in the API process, once
in the Celery worker) because each process independently forgot to set
it and silently fell back to a local ./mlruns file store. This module
exists so there is exactly one place that does it, called by every
process that talks to MLflow: the API (app/main.py), the Celery worker
(workers/celery_app.py), and the training CLI (ml/training/train.py).

configure_mlflow() is idempotent — safe to call multiple times (e.g. at
both module import and inside a function) since it's just two mlflow.*
setter calls, not a connection that needs to be opened once.
"""

import mlflow

from app.core.config import get_settings


def configure_mlflow() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def assert_mlflow_configured() -> None:
    """Raise if the current process's mlflow tracking URI doesn't match
    the configured one — call this at process startup so a
    configuration regression fails loudly instead of silently writing
    to the wrong (local file store) backend again.
    """
    settings = get_settings()
    actual = mlflow.get_tracking_uri()
    if actual != settings.mlflow_tracking_uri:
        raise RuntimeError(
            f"MLflow tracking URI mismatch: expected {settings.mlflow_tracking_uri!r} "
            f"(from MLFLOW_TRACKING_URI) but mlflow reports {actual!r}. This usually means "
            "configure_mlflow() was not called before this check, or something reset the "
            "tracking URI afterward — every process must call configure_mlflow() at startup."
        )
