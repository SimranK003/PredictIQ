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
    model_config = {"protected_namespaces": ()}

    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    n_records: int | None
    model_version: str | None
    result: dict | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
