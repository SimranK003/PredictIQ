"""Async training job submission.

POST /train returns as soon as the job is queued — training itself runs
in a Celery worker (see app/services/training.py, workers/tasks.py), not
inside this request.

Duplicate-job policy (documented, not silently decided): if an ACTIVE
(queued or running) job already exists for the same dataset_id with the
exact same resolved config+algorithms, that existing job is returned
instead of creating a new one. This guards against the common accidental
case (a double-click, a retried request) without distributed locking —
a small race between the existence check and the insert is acceptable
here because, unlike the production/previous registry invariant, two
near-simultaneous duplicate training jobs are low-stakes (wasted compute,
not corrupted state) and Postgres isn't asked to prevent it structurally.
Two different configs against the same dataset, or a resubmission after
the earlier job finished (completed or failed), are treated as
legitimately new jobs.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.schemas.train import TrainRequest, TrainResponse
from db.models import Dataset, Job, JobStatus, JobType
from db.session import get_db

router = APIRouter(tags=["training"])
logger = get_logger(__name__)

_ACTIVE_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)


def _find_duplicate_active_job(db: Session, dataset_id, config: dict, algorithms) -> Job | None:
    candidates = (
        db.query(Job)
        .filter(
            Job.job_type == JobType.TRAIN,
            Job.dataset_id == dataset_id,
            Job.status.in_(_ACTIVE_STATUSES),
        )
        .all()
    )
    for job in candidates:
        if job.payload.get("config") == config and job.payload.get("algorithms") == algorithms:
            return job
    return None


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_training_job(request: TrainRequest, db: Session = Depends(get_db)) -> TrainResponse:
    dataset = db.get(Dataset, request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    if not dataset.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Dataset {dataset.id} failed validation at ingestion and cannot be used "
                "for training. Check its quality_report via GET /datasets/{id}."
            ),
        )

    config_overrides = request.resolved_config_overrides()
    algorithms = request.algorithms  # None means "all", stored as null and resolved at run time

    duplicate = _find_duplicate_active_job(db, dataset.id, config_overrides, algorithms)
    if duplicate is not None:
        logger.info(
            "training_job_duplicate_returned",
            extra={"job_id": str(duplicate.id), "dataset_id": str(dataset.id)},
        )
        return TrainResponse(
            job_id=duplicate.id,
            dataset_id=dataset.id,
            status=duplicate.status,
            created_at=duplicate.created_at,
        )

    job = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.QUEUED,
        dataset_id=dataset.id,
        payload={"config": config_overrides, "algorithms": algorithms},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(
        "training_job_created",
        extra={"job_id": str(job.id), "dataset_id": str(dataset.id), "config": config_overrides},
    )

    from workers.tasks import process_training_job_task

    try:
        async_result = process_training_job_task.delay(str(job.id))
        job.celery_task_id = async_result.id
        db.commit()
    except Exception as exc:
        logger.exception("training_job_dispatch_failed", extra={"job_id": str(job.id)})
        job.status = JobStatus.FAILED
        job.error_message = f"Could not dispatch to the worker queue: {exc}"[:2000]
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Training job {job.id} was created but could not be dispatched to the "
                "worker queue (broker unavailable). Check GET /jobs/{job_id} for status."
            ),
        ) from exc

    return TrainResponse(
        job_id=job.id, dataset_id=dataset.id, status=job.status, created_at=job.created_at
    )
