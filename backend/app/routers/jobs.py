"""Async job status endpoint."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.job import JobOut
from db.models import Job, JobType, ModelVersionRecord
from db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

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
