"""FastAPI application entrypoint."""

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.routers import datasets, health, models

configure_logging()
settings = get_settings()

# Required so the registry's artifact-existence check (and, in Phase 4,
# the production model loader) actually reach the configured tracking
# server rather than silently falling back to mlflow's default local
# ./mlruns file store.
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

app = FastAPI(
    title="PredictIQ",
    description="End-to-end ML prediction & monitoring platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(models.router)
