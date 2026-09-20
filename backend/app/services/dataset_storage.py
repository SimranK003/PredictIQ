"""Dataset file storage: local filesystem (default) or S3-compatible
object storage, selected by DATASET_STORAGE_BACKEND.

Why this exists (Phase 10): local dev and Docker Compose share one
filesystem/volume between the backend and worker containers, so a plain
Path.write_bytes()/read_bytes() has always been enough (see git history
of app/services/ingestion.py and app/services/training.py before this
module). A real cloud deployment where backend and worker are separate
services typically has no such shared disk — see README's Production
deployment section for why. This module is the single place that
branches on backend so ingestion.py, training.py, and
ml/training/train.py don't each need their own local-vs-S3 logic.

Local behavior is completely unchanged (same paths, same
Path.write_bytes/read_bytes calls) unless DATASET_STORAGE_BACKEND=s3 is
explicitly set — this module is additive, not a rewrite of the existing
storage behavior.
"""

import io
import uuid
from pathlib import Path

import pandas as pd

from app.core.config import get_settings

_S3_URI_PREFIX = "s3://"


def _is_s3_path(storage_path: str) -> bool:
    return storage_path.startswith(_S3_URI_PREFIX)


def _parse_s3_uri(storage_path: str) -> tuple[str, str]:
    without_prefix = storage_path[len(_S3_URI_PREFIX) :]
    bucket, _, key = without_prefix.partition("/")
    return bucket, key


def _s3_client():
    import boto3  # local import: only needed when the s3 backend is actually used

    settings = get_settings()
    return boto3.client(
        "s3",
        region_name=settings.dataset_storage_s3_region or None,
        endpoint_url=settings.dataset_storage_s3_endpoint_url or None,
    )


def save_dataset_bytes(dataset_id: uuid.UUID, raw_bytes: bytes) -> str:
    """Persist a raw uploaded CSV and return the value to store as
    Dataset.storage_path — a local filesystem path, or an s3://
    URI, depending on DATASET_STORAGE_BACKEND.
    """
    settings = get_settings()
    if settings.dataset_storage_backend == "s3":
        bucket = settings.dataset_storage_s3_bucket
        key = f"{settings.dataset_storage_s3_prefix.rstrip('/')}/{dataset_id}.csv"
        _s3_client().put_object(Bucket=bucket, Key=key, Body=raw_bytes)
        return f"{_S3_URI_PREFIX}{bucket}/{key}"

    directory = Path(settings.dataset_storage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{dataset_id}.csv"
    file_path.write_bytes(raw_bytes)
    return str(file_path)


def dataset_file_exists(storage_path: str) -> bool:
    if _is_s3_path(storage_path):
        bucket, key = _parse_s3_uri(storage_path)
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return Path(storage_path).exists()


def read_dataset_bytes(storage_path: str) -> bytes:
    if _is_s3_path(storage_path):
        bucket, key = _parse_s3_uri(storage_path)
        response = _s3_client().get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    return Path(storage_path).read_bytes()


def read_dataset_dataframe(storage_path: str) -> pd.DataFrame:
    """Same dtype=str contract as ml/training/data.py's
    load_raw_dataframe (delegated to directly for the local case, so
    that existing, already-tested function stays the one implementation
    of "how a CSV becomes a raw dataframe" for local paths).
    """
    if _is_s3_path(storage_path):
        return pd.read_csv(io.BytesIO(read_dataset_bytes(storage_path)), dtype=str)

    from ml.training.data import load_raw_dataframe

    return load_raw_dataframe(storage_path)
