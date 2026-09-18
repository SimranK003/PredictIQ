"""FastAPI application entrypoint."""

import mlflow
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.routers import datasets, health, jobs, models, monitoring, predictions

configure_logging()
settings = get_settings()
logger = get_logger(__name__)

# Required so the registry's artifact-existence check and the production
# model loader actually reach the configured tracking server rather than
# silently falling back to mlflow's default local ./mlruns file store.
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
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: never leak a Python traceback to a client.

    The real exception is logged server-side (with request_id for
    correlation); the client gets a generic message and a 500.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "unhandled_exception",
        extra={"request_id": request_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred.", "request_id": request_id},
    )


app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(jobs.router)
app.include_router(monitoring.router)
