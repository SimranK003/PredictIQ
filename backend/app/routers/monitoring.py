"""Observability endpoints: Prometheus metrics, drift detection, model
usage, training job stats, and the aggregate summary.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from db.session import get_db
from monitoring.drift import resolve_drift_report
from monitoring.job_stats import get_training_job_stats
from monitoring.model_usage import get_model_version_usage
from monitoring.summary import get_monitoring_summary

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/monitoring/drift")
def drift(
    model_version_id: uuid.UUID | None = Query(
        None, description="Defaults to the current production model."
    ),
    window: str | None = Query(None, description="One of: 24h, 7d, 30d. Omit for all-time."),
    feature: str | None = Query(None, description="Restrict to one feature; omit for all."),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return resolve_drift_report(
            db, model_version_id=model_version_id, window=window, feature=feature
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/monitoring/model-usage")
def model_usage(db: Session = Depends(get_db)) -> list[dict]:
    return get_model_version_usage(db)


@router.get("/monitoring/jobs")
def job_stats(db: Session = Depends(get_db)) -> dict:
    return get_training_job_stats(db)


@router.get("/monitoring/summary")
def summary(db: Session = Depends(get_db)) -> dict:
    return get_monitoring_summary(db)
