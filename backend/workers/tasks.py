"""Celery tasks. Currently one: process a queued batch-prediction job."""

from app.core.logging import configure_logging, get_logger
from app.services.prediction import execute_batch_prediction
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
