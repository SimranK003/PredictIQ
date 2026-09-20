"""Dataset ingestion endpoints."""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.exceptions import (
    DatasetValidationError,
    FileTooLargeError,
    MalformedCSVError,
    UnsupportedFileTypeError,
)
from app.core.security import get_current_user
from app.schemas.dataset import DatasetOut, DatasetSummaryOut
from app.services.ingestion import ingest_csv_upload
from db.models import Dataset
from db.session import get_db

# Every dataset endpoint requires an authenticated session — enforced
# here at the router level (dependencies=[...]) rather than per-endpoint
# so a new route added later can't accidentally ship unprotected.
router = APIRouter(prefix="/datasets", tags=["datasets"], dependencies=[Depends(get_current_user)])


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)) -> Dataset:
    raw_bytes = await file.read()
    try:
        return ingest_csv_upload(db, filename=file.filename or "upload.csv", raw_bytes=raw_bytes)
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "quality_report": exc.report},
        ) from exc
    except (UnsupportedFileTypeError, MalformedCSVError, FileTooLargeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("", response_model=list[DatasetSummaryOut])
def list_datasets(db: Session = Depends(get_db)) -> list[Dataset]:
    return db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return dataset
