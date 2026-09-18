"""Celery application for async jobs.

Currently just batch prediction (see tasks.py) — training/ingestion jobs
becoming async is Phase 5's scope, not rebuilt here.

To actually process queued jobs, a worker process must be running:
    celery -A workers.celery_app worker --loglevel=info
Without a worker running, jobs enqueued via .delay() sit in Redis at
QUEUED forever — this is correct, honest behavior (not a bug) for a
queue with no consumer, and is called out in the API docs/README rather
than silently pretending async processing happened.
"""

from celery import Celery

from app.core.config import get_settings
from app.core.mlflow_config import assert_mlflow_configured, configure_mlflow

settings = get_settings()

# Each process configures mlflow's tracking URI independently (it's
# process-global state, not shared with the API process) — otherwise
# this worker silently falls back to a local ./mlruns file store and
# can't find any real run's artifact. See app/core/mlflow_config.py —
# this exact bug was hit here once already (Phase 4) before being fixed.
configure_mlflow()
assert_mlflow_configured()

celery_app = Celery(
    "predictiq",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Celery hijacks the root logger by default (worker_hijack_root_logger
    # defaults to True), silently discarding our JSON logging setup
    # (app/core/logging.py) and replacing it with its own plain-text
    # formatter — found while verifying that structured logs actually
    # appear in a real worker process, not assumed to work because the
    # code looked right. False here means Celery leaves the root logger
    # alone, so configure_logging() (called in workers/tasks.py) is what
    # actually determines the worker's log format.
    worker_hijack_root_logger=False,
)
