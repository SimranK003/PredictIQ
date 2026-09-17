"""Dataset ingestion service.

Orchestrates: file-type/size checks -> CSV parse -> schema validation ->
persistence (raw file to disk, metadata + quality report to Postgres).

Kept separate from the router so it's independently unit-testable and so
the same logic can be reused by a future CLI or batch-ingestion job.
"""

import io
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    DatasetValidationError,
    FileTooLargeError,
    MalformedCSVError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from db.models import Dataset
from ml.schema import CHURN_SCHEMA, DatasetSchema
from ml.validation import validate_dataset

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".csv"}


def _storage_dir() -> Path:
    path = Path(get_settings().dataset_storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ingest_csv_upload(
    db: Session,
    *,
    filename: str,
    raw_bytes: bytes,
    schema: DatasetSchema = CHURN_SCHEMA,
) -> Dataset:
    """Validate an uploaded CSV and persist it if valid.

    Raises DatasetValidationError (with the report attached) if the
    dataset fails hard checks — the caller should map that to HTTP 422.
    """
    _check_file_type(filename)
    _check_file_size(raw_bytes)
    df = _parse_csv(raw_bytes)

    report = validate_dataset(df, schema)

    dataset_id = uuid.uuid4()
    storage_path = _storage_dir() / f"{dataset_id}.csv"

    # Persist the raw file regardless of validity so a rejected upload can
    # still be inspected/debugged, but only via the quality report — we
    # don't silently proceed to training with it.
    storage_path.write_bytes(raw_bytes)

    dataset = Dataset(
        id=dataset_id,
        filename=filename,
        schema_name=schema.name,
        storage_path=str(storage_path),
        n_rows=report.n_rows,
        n_columns=report.n_columns,
        is_valid=report.is_valid,
        quality_report=report.to_dict(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    logger.info(
        "dataset_ingested",
        extra={"dataset_id": str(dataset_id), "is_valid": report.is_valid, "n_rows": report.n_rows},
    )

    if not report.is_valid:
        raise DatasetValidationError(
            f"Dataset '{filename}' failed validation.", report=dataset.quality_report
        )

    return dataset


def _check_file_type(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Only CSV files are accepted."
        )


def _check_file_size(raw_bytes: bytes) -> None:
    max_bytes = get_settings().max_upload_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise FileTooLargeError(
            f"File size {len(raw_bytes) / (1024 * 1024):.1f}MB exceeds the "
            f"{get_settings().max_upload_size_mb}MB limit."
        )
    if len(raw_bytes) == 0:
        raise MalformedCSVError("Uploaded file is empty.")


def _parse_csv(raw_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), dtype=str)
    except Exception as exc:  # pandas raises several distinct error types
        raise MalformedCSVError(f"Could not parse file as CSV: {exc}") from exc
