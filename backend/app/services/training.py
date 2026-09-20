"""Async training job execution: orchestrates the existing ML modules
(ml/training/train.py) against a queued Job row.

Does not reimplement training — train_and_register() already is "dataset
-> preprocessing -> models -> MLflow -> registered candidates"; this
module's job is purely the Job-lifecycle bookkeeping around that call
(status transitions, dataset re-verification, safe error capture) shared
by both the sync-inline test path and the real Celery task.

Retry classification (see workers/tasks.py): exceptions raised by this
function's *dataset loading/verification and train_and_register* step
propagate uncaught when they're TRANSIENT_EXCEPTIONS (infra: DB/MLflow
connectivity) so Celery can retry the whole job; anything else is treated
as a deterministic ML/data failure and is caught here, recorded on the
Job as FAILED, and never retried automatically — retrying a bad dataset
or a config that makes XGBoost blow up would just fail again identically.
"""

import hashlib
from datetime import UTC, datetime

import requests
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.exceptions import PredictIQError
from app.core.logging import get_logger
from app.services.dataset_storage import dataset_file_exists, read_dataset_bytes
from db.models import Dataset, Job, JobStatus, ModelVersionRecord
from ml.training.config import TrainingConfig
from ml.training.models import ALGORITHMS
from ml.training.train import train_and_register

logger = get_logger(__name__)

# Infra failures worth retrying: a dropped DB connection or an
# unreachable MLflow tracking server (its REST client raises requests'
# connection/timeout errors when it can't reach the server at all,
# before ever getting an HTTP response to wrap in an MlflowException).
TRANSIENT_EXCEPTIONS = (
    OperationalError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class TrainingExecutionError(PredictIQError):
    """Deterministic training/data failure — never worth retrying."""


def _verify_dataset_for_training(db: Session, dataset_id) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise TrainingExecutionError(f"Dataset {dataset_id} not found.")
    if not dataset.is_valid:
        raise TrainingExecutionError(
            f"Dataset {dataset_id} failed validation at ingestion and cannot be used "
            "for training."
        )

    if not dataset_file_exists(dataset.storage_path):
        raise TrainingExecutionError(f"Dataset file is missing: {dataset.storage_path}")

    actual_hash = hashlib.sha256(read_dataset_bytes(dataset.storage_path)).hexdigest()
    if actual_hash != dataset.content_hash:
        raise TrainingExecutionError(
            f"Dataset {dataset_id} content hash mismatch: recorded {dataset.content_hash}, "
            f"file on disk now hashes to {actual_hash}. The file may have been modified "
            "since ingestion — refusing to train on it."
        )

    return dataset


def _safe_error_message(exc: Exception) -> str:
    # str(exc) is a clean one-line message by construction (Python never
    # includes the traceback there) — this is the same pattern used for
    # batch prediction failures in app/services/prediction.py.
    return str(exc)[:2000]


def execute_training_job(db: Session, job_id) -> None:
    job = db.get(Job, job_id)
    if job is None:
        logger.error("training_job_not_found", extra={"job_id": str(job_id)})
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC)
    db.commit()
    logger.info(
        "training_job_started", extra={"job_id": str(job_id), "dataset_id": str(job.dataset_id)}
    )

    try:
        dataset = _verify_dataset_for_training(db, job.dataset_id)
        logger.info(
            "training_dataset_loaded",
            extra={"job_id": str(job_id), "dataset_id": str(dataset.id), "n_rows": dataset.n_rows},
        )

        config = TrainingConfig(**job.payload.get("config", {}))
        algorithms = job.payload.get("algorithms") or list(ALGORITHMS)

        candidates = train_and_register(
            db,
            dataset=dataset,
            config=config,
            algorithms=algorithms,
            training_job_id=job.id,
        )
    except TRANSIENT_EXCEPTIONS:
        # Let Celery's retry logic handle it (see workers/tasks.py) — the
        # job stays RUNNING; a retry re-enters this function from the top.
        raise
    except Exception as exc:
        logger.exception("training_job_failed", extra={"job_id": str(job_id)})
        db.rollback()

        # Query, don't trust in-memory state: recovers exactly what was
        # actually persisted before the failure, regardless of where in
        # the loop it happened (see module docstring — no orphans, since
        # each candidate is only ever committed after its MLflow run and
        # artifact were already fully logged).
        partial_ids = [
            str(row.id)
            for row in db.query(ModelVersionRecord.id)
            .filter(ModelVersionRecord.training_job_id == job_id)
            .all()
        ]

        job = db.get(Job, job_id)
        job.status = JobStatus.FAILED
        job.error_message = _safe_error_message(exc)
        job.result = {"partial_candidate_model_version_ids": partial_ids} if partial_ids else None
        job.finished_at = datetime.now(UTC)
        db.commit()
        return

    best = max(candidates, key=lambda c: c.metrics["roc_auc"])
    job = db.get(Job, job_id)
    job.status = JobStatus.COMPLETED
    job.result = {
        "candidate_model_version_ids": [str(c.id) for c in candidates],
        "algorithms_trained": [c.algorithm for c in candidates],
        "best_algorithm": best.algorithm,
        "best_model_version_id": str(best.id),
        "best_test_roc_auc": best.metrics["roc_auc"],
    }
    job.finished_at = datetime.now(UTC)
    db.commit()
    logger.info(
        "training_job_completed",
        extra={
            "job_id": str(job_id),
            "dataset_id": str(job.dataset_id),
            "n_candidates": len(candidates),
            "best_algorithm": best.algorithm,
            "best_model_version_id": str(best.id),
        },
    )
