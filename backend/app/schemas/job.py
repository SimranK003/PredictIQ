"""Pydantic schemas for the job status API.

Deliberately excludes payload.records from the response — echoing back
the full input batch on every status poll is unnecessary bloat, and that
data is already queryable per-prediction via GET /predictions.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel

from db.models import JobStatus, JobType


class JobOut(BaseModel):
    """Different job types populate different subsets of these fields:
    n_records/model_version are batch-prediction-specific; dataset_id/
    model_version_ids are training-specific. Unused fields are null.
    """

    model_config = {"protected_namespaces": ()}

    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    dataset_id: uuid.UUID | None
    n_records: int | None
    model_version: str | None
    model_version_ids: list[uuid.UUID] | None
    result: dict | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class PaginatedJobsOut(BaseModel):
    items: list[JobOut]
    total: int
    limit: int
    offset: int
