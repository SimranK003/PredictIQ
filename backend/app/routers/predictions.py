"""Inference endpoints: single/batch prediction and prediction history."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.core.middleware import get_request_id
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnFeaturesIn,
    PaginatedPredictionsOut,
    PredictionDetailOut,
    PredictionOut,
)
from app.services.model_loader import get_production_pipeline
from app.services.prediction import (
    PredictionExecutionError,
    execute_batch_prediction,
    run_single_prediction,
)
from db.models import Job, JobStatus, JobType, ModelVersionRecord, Prediction
from db.session import get_db
from monitoring.metrics import BATCH_PREDICTIONS_TOTAL

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionOut, status_code=status.HTTP_200_OK)
def predict(
    features: ChurnFeaturesIn,
    db: Session = Depends(get_db),
    request_id: uuid.UUID = Depends(get_request_id),
) -> PredictionOut:
    try:
        result = run_single_prediction(db, features.model_dump(), request_id=request_id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except PredictionExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    row = result.prediction_row
    return PredictionOut(
        prediction_id=row.id,
        prediction=row.prediction,
        probability=row.probability,
        model_version=result.model_version.version_label,
        model_version_id=result.model_version.id,
        timestamp=row.created_at,
        request_id=row.request_id,
    )


@router.post(
    "/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_202_ACCEPTED
)
def predict_batch(
    payload: BatchPredictionRequest, db: Session = Depends(get_db)
) -> BatchPredictionResponse:
    settings = get_settings()
    n_records = len(payload.records)

    if n_records > settings.batch_prediction_max_records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch of {n_records} records exceeds the maximum of "
                f"{settings.batch_prediction_max_records}."
            ),
        )

    try:
        model_version, _ = get_production_pipeline(db)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    job = Job(
        job_type=JobType.BATCH_PREDICT,
        status=JobStatus.QUEUED,
        payload={
            "records": [r.model_dump() for r in payload.records],
            "model_version_id": str(model_version.id),
        },
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    if n_records <= settings.batch_prediction_sync_max_records:
        BATCH_PREDICTIONS_TOTAL.labels(mode="sync").inc()
        execute_batch_prediction(db, job.id)
        db.refresh(job)
    else:
        from workers.tasks import process_batch_prediction_task

        BATCH_PREDICTIONS_TOTAL.labels(mode="async").inc()
        process_batch_prediction_task.delay(str(job.id))

    return BatchPredictionResponse(
        job_id=job.id,
        status=job.status,
        n_records=n_records,
        model_version=model_version.version_label,
    )


@router.get("/predictions", response_model=PaginatedPredictionsOut)
def list_predictions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    model_version_id: uuid.UUID | None = Query(None),
    start_date: datetime | None = Query(None, description="Inclusive lower bound on created_at."),
    end_date: datetime | None = Query(None, description="Inclusive upper bound on created_at."),
    db: Session = Depends(get_db),
) -> PaginatedPredictionsOut:
    query = db.query(Prediction)
    if model_version_id is not None:
        query = query.filter(Prediction.model_version_id == model_version_id)
    if start_date is not None:
        query = query.filter(Prediction.created_at >= start_date)
    if end_date is not None:
        query = query.filter(Prediction.created_at <= end_date)

    total = query.count()
    items = query.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()

    return PaginatedPredictionsOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/predictions/{prediction_id}", response_model=PredictionDetailOut)
def get_prediction(prediction_id: uuid.UUID, db: Session = Depends(get_db)) -> PredictionDetailOut:
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")

    model_version = db.get(ModelVersionRecord, prediction.model_version_id)
    return PredictionDetailOut(
        id=prediction.id,
        request_id=prediction.request_id,
        model_version_id=prediction.model_version_id,
        model_version_label=model_version.version_label,
        algorithm=model_version.algorithm,
        input_features=prediction.input_features,
        prediction=prediction.prediction,
        probability=prediction.probability,
        latency_ms=prediction.latency_ms,
        created_at=prediction.created_at,
    )
