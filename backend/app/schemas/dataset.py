"""Pydantic response models for the dataset ingestion API."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class ValidationIssueOut(BaseModel):
    check: str
    severity: str
    message: str
    details: dict = {}


class DataQualityReportOut(BaseModel):
    n_rows: int
    n_columns: int
    is_valid: bool
    duplicate_row_count: int
    column_null_counts: dict[str, int]
    class_balance: dict[str, int] | None
    issues: list[ValidationIssueOut]


class DatasetOut(BaseModel):
    id: uuid.UUID
    filename: str
    schema_name: str
    n_rows: int
    n_columns: int
    is_valid: bool
    uploaded_at: datetime
    quality_report: DataQualityReportOut

    model_config = {"from_attributes": True}


class DatasetSummaryOut(BaseModel):
    id: uuid.UUID
    filename: str
    schema_name: str
    n_rows: int
    is_valid: bool
    uploaded_at: datetime

    model_config = {"from_attributes": True}
