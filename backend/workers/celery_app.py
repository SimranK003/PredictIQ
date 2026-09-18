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

import mlflow
from celery import Celery

from app.core.config import get_settings

settings = get_settings()

# Each process configures mlflow's tracking URI independently (it's
# process-global state, not shared with the API process) — otherwise
# this worker silently falls back to a local ./mlruns file store and
# can't find any real run's artifact. Same class of bug already fixed
# once in app/main.py; worth fixing here explicitly rather than
# assuming "it's set somewhere" carries across process boundaries.
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

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
)
