"""Liveness/readiness endpoint.

Three-tier status, matching what each dependency actually means for the
service:
- "unhealthy" (503): the database is unreachable — the API can't do
  almost anything without it, including this health check's own query.
- "degraded" (200): the API is up and most endpoints work, but something
  that matters for serving predictions is off — no production model
  registered, or Redis (needed for large async batches) unreachable.
- "ok" (200): everything checked is available.

Never includes connection strings, credentials, or other internals —
only ok/unavailable per dependency.
"""

import redis
from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.registry import get_production_model
from db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(response: Response, db: Session = Depends(get_db)) -> dict:
    settings = get_settings()

    try:
        db.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception:
        database_status = "unavailable"

    try:
        redis_client = redis.from_url(settings.redis_url, socket_connect_timeout=1)
        redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    if database_status == "ok":
        production = get_production_model(db)
        production_model = {
            "available": production is not None,
            "version": production.version_label if production else None,
            "algorithm": production.algorithm if production else None,
        }
    else:
        production_model = {"available": False, "version": None, "algorithm": None}

    if database_status == "unavailable":
        overall = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif redis_status == "unavailable" or not production_model["available"]:
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "status": overall,
        "database": database_status,
        "redis": redis_status,
        "production_model": production_model,
    }
