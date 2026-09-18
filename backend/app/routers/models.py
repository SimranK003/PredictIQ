"""Model registry API: browse versions, compare candidates, promote, roll back."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.exceptions import InvalidPromotionError, NotFoundError, PromotionConflictError
from app.schemas.model import (
    ModelMetricsOut,
    ModelVersionOut,
    ModelVersionSummaryOut,
    PromoteRequest,
)
from app.services.model_loader import invalidate_production_model_cache
from app.services.registry import get_production_model, promote_model, rollback_model
from db.models import ModelStage, ModelVersionRecord
from db.session import get_db

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionSummaryOut])
def list_models(
    stage: ModelStage | None = Query(None, description="Filter by lifecycle stage."),
    db: Session = Depends(get_db),
) -> list[ModelVersionRecord]:
    query = db.query(ModelVersionRecord)
    if stage is not None:
        query = query.filter(ModelVersionRecord.stage == stage)
    return query.order_by(ModelVersionRecord.created_at.desc()).all()


@router.get("/production", response_model=ModelVersionOut)
def get_current_production_model(db: Session = Depends(get_db)) -> ModelVersionRecord:
    model_version = get_production_model(db)
    if model_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No production model is registered yet."
        )
    return model_version


@router.get("/{model_id}", response_model=ModelVersionOut)
def get_model(model_id: uuid.UUID, db: Session = Depends(get_db)) -> ModelVersionRecord:
    model_version = db.get(ModelVersionRecord, model_id)
    if model_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found."
        )
    return model_version


@router.get("/{model_id}/metrics", response_model=ModelMetricsOut)
def get_model_metrics(model_id: uuid.UUID, db: Session = Depends(get_db)) -> ModelVersionRecord:
    model_version = db.get(ModelVersionRecord, model_id)
    if model_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found."
        )
    return model_version


@router.post("/promote", response_model=ModelVersionOut, status_code=status.HTTP_200_OK)
def promote(request: PromoteRequest, db: Session = Depends(get_db)) -> ModelVersionRecord:
    try:
        promoted = promote_model(db, request.candidate_id, triggered_by="api")
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidPromotionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except PromotionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    # Invalidate in this process's cache so the very next inference request
    # in this worker picks up the new production model. Other worker
    # processes self-heal on their next request (see model_loader.py).
    invalidate_production_model_cache()
    return promoted


@router.post("/rollback", response_model=ModelVersionOut, status_code=status.HTTP_200_OK)
def rollback(db: Session = Depends(get_db)) -> ModelVersionRecord:
    try:
        restored = rollback_model(db, triggered_by="api")
    except InvalidPromotionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except PromotionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    invalidate_production_model_cache()
    return restored
