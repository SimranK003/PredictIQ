"""Async job status endpoints.

GET /jobs (list) is a Phase 7 addition — the /jobs dashboard page needs
to enumerate jobs, and only a single-job lookup existed. It mirrors the
exact list+detail pattern /datasets and /models already use; nothing
about the underlying data model or job lifecycle changes.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.schemas.job import JobOut, PaginatedJobsOut
from db.models import Job, JobStatus, JobType, ModelVersionRecord
from db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(get_current_user)])


def _build_job_out(db: Session, job: Job) -> JobOut:
    n_records = None
    model_version = None
    model_version_ids = None

    if job.job_type == JobType.BATCH_PREDICT and isinstance(job.payload, dict):
        if "records" in job.payload:
            n_records = len(job.payload["records"])
        if job.payload.get("model_version_id"):
            pinned = db.get(ModelVersionRecord, job.payload["model_version_id"])
            model_version = pinned.version_label if pinned else None

    if job.job_type == JobType.TRAIN:
        candidates = (
            db.query(ModelVersionRecord.id)
            .filter(ModelVersionRecord.training_job_id == job.id)
            .all()
        )
        if candidates:
            model_version_ids = [row.id for row in candidates]

    return JobOut(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        dataset_id=job.dataset_id,
        n_records=n_records,
        model_version=model_version,
        model_version_ids=model_version_ids,
        result=job.result,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("", response_model=PaginatedJobsOut)
def list_jobs(
    job_type: JobType | None = Query(None),
    status_filter: JobStatus | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedJobsOut:
    query = db.query(Job)
    if job_type is not None:
        query = query.filter(Job.job_type == job_type)
    if status_filter is not None:
        query = query.filter(Job.status == status_filter)

    total = query.count()
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()

    return PaginatedJobsOut(
        items=[_build_job_out(db, j) for j in jobs], total=total, limit=limit, offset=offset
    )


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _build_job_out(db, job)
