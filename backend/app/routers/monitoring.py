"""Observability endpoints: Prometheus metrics and the drift foundation."""

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from db.session import get_db
from monitoring.drift import resolve_production_drift_status

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/monitoring/drift")
def drift(db: Session = Depends(get_db)) -> dict:
    return resolve_production_drift_status(db)
