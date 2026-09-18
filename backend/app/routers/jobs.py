"""Async job status endpoint."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.job import JobOut
from db.models import Job, ModelVersionRecord
from db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    n_records = None
    if isinstance(job.payload, dict) and "records" in job.payload:
        n_records = len(job.payload["records"])

    model_version = None
    if isinstance(job.payload, dict) and job.payload.get("model_version_id"):
        pinned = db.get(ModelVersionRecord, job.payload["model_version_id"])
        model_version = pinned.version_label if pinned else None

    return JobOut(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        n_records=n_records,
        model_version=model_version,
        result=job.result,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
