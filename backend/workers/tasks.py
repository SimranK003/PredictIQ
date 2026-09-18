"""Celery tasks: batch prediction and async model training."""

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.prediction import execute_batch_prediction
from app.services.training import TRANSIENT_EXCEPTIONS, execute_training_job
from db.session import SessionLocal
from workers.celery_app import celery_app

configure_logging()
logger = get_logger(__name__)


@celery_app.task(name="predictiq.process_batch_prediction")
def process_batch_prediction_task(job_id: str) -> None:
    db = SessionLocal()
    try:
        execute_batch_prediction(db, job_id)
    finally:
        db.close()


@celery_app.task(bind=True, name="predictiq.process_training_job")
def process_training_job_task(self, job_id: str) -> None:
    """Retries only on transient infra errors (DB/MLflow connectivity) —
    see app/services/training.py's TRANSIENT_EXCEPTIONS and module
    docstring. A deterministic data/training failure is caught inside
    execute_training_job itself, recorded as JobStatus.FAILED, and
    returns normally from this task (no retry, no re-raise).
    """
    settings = get_settings()
    db = SessionLocal()
    try:
        execute_training_job(db, job_id)
    except TRANSIENT_EXCEPTIONS as exc:
        logger.warning(
            "training_job_transient_failure_retrying",
            extra={"job_id": job_id, "attempt": self.request.retries, "error": str(exc)},
        )
        raise self.retry(
            exc=exc,
            max_retries=settings.training_job_max_retries,
            countdown=settings.training_job_retry_delay_seconds,
        ) from exc
    finally:
        db.close()
