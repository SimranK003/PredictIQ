"""Efficient, self-healing loader for the current production model.

Phase 4's inference endpoints will call get_production_pipeline(db) on
every request. This module makes that cheap: it keeps the last-loaded
sklearn Pipeline in memory and only calls mlflow.sklearn.load_model()
again when the production model_version_id actually changes.

Two invalidation paths, deliberately both present:
1. Explicit: app/routers/models.py calls invalidate() right after a
   successful promote/rollback, so the *same process* picks up the new
   model on its very next request.
2. Self-healing by id comparison: every get() call re-reads which model
   is production from Postgres (one cheap indexed lookup) and reloads if
   the id changed. This is what protects against staleness in a
   multi-worker deployment (gunicorn/uvicorn with >1 worker process) —
   explicit invalidation only reaches the worker that handled the
   promote request, but every worker's next request will see the new
   production id and reload on its own.

Known limitation (documented, not silently accepted): with N worker
processes, a promotion is only *guaranteed* visible to all of them after
each has served one more request — there is no cross-process push
invalidation (that would need pub/sub, e.g. Redis, which is more
infrastructure than a single-model portfolio deployment needs today).
"""

import threading
import uuid
from typing import Any

import mlflow.sklearn
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.services.registry import get_production_model
from db.models import ModelVersionRecord


class ProductionModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_version_id: uuid.UUID | None = None
        self._pipeline: Any | None = None

    def get(self, db: Session) -> tuple[ModelVersionRecord, Any]:
        current = get_production_model(db)
        if current is None:
            raise NotFoundError("No production model is currently registered.")

        with self._lock:
            if self._model_version_id != current.id:
                self._pipeline = mlflow.sklearn.load_model(current.artifact_uri)
                self._model_version_id = current.id
            return current, self._pipeline

    def invalidate(self) -> None:
        with self._lock:
            self._model_version_id = None
            self._pipeline = None

    @property
    def loaded_model_version_id(self) -> uuid.UUID | None:
        return self._model_version_id


_cache = ProductionModelCache()


def get_production_pipeline(db: Session) -> tuple[ModelVersionRecord, Any]:
    """Returns (production model_version record, loaded sklearn Pipeline).

    Raises NotFoundError if no model is currently in production.
    """
    return _cache.get(db)


def invalidate_production_model_cache() -> None:
    _cache.invalidate()
