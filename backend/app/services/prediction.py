"""Prediction execution: resolve the production pipeline, run raw
features straight through it (no separate preprocessing step — the
serialized Pipeline already includes feature engineering + preprocessing
from training, see ml/preprocessing.py), and persist the result.

Shared by both the synchronous /predict(/batch) path and the async
Celery batch task (workers/tasks.py) so there's exactly one place that
implements "how a prediction gets made and recorded."
"""

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError, PredictIQError
from app.core.logging import get_logger
from app.services.model_loader import get_production_pipeline, load_pipeline_for_model_version
from db.models import Job, JobStatus, ModelVersionRecord, Prediction
from ml.schema import CHURN_SCHEMA
from monitoring.metrics import PREDICTION_ERRORS_TOTAL, PREDICTIONS_TOTAL

logger = get_logger(__name__)


class PredictionExecutionError(PredictIQError):
    """The pipeline itself raised while predicting on otherwise-valid input."""


@dataclass
class PredictionResult:
    prediction_row: Prediction
    model_version: ModelVersionRecord


def _build_feature_dataframe(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with exactly the trained feature columns, in the
    dict-derived order given — column order doesn't matter to the
    ColumnTransformer (it selects by name), but every record must have
    every expected column present so a missing key surfaces immediately
    rather than as a confusing downstream KeyError.
    """
    return pd.DataFrame(records, columns=list(CHURN_SCHEMA.all_feature_columns))


def _labels_and_probabilities(pipeline: Any, df: pd.DataFrame) -> tuple[list[str], list[float]]:
    raw_predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]
    labels = [
        CHURN_SCHEMA.positive_label if p == 1 else CHURN_SCHEMA.negative_label
        for p in raw_predictions
    ]
    return labels, [float(p) for p in probabilities]


def run_single_prediction(
    db: Session, features: dict, *, request_id: uuid.UUID
) -> PredictionResult:
    """Resolve production, predict on one record, persist, and return it.

    Raises NotFoundError if no production model is registered, or
    PredictionExecutionError if the pipeline itself fails on this input.
    """
    try:
        model_version, pipeline = get_production_pipeline(db)
    except NotFoundError:
        PREDICTION_ERRORS_TOTAL.labels(reason="model_unavailable").inc()
        raise

    df = _build_feature_dataframe([features])

    start = time.perf_counter()
    try:
        labels, probabilities = _labels_and_probabilities(pipeline, df)
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.labels(reason="prediction_exception").inc()
        logger.exception(
            "prediction_pipeline_failed",
            extra={"request_id": str(request_id), "model_version_id": str(model_version.id)},
        )
        raise PredictionExecutionError("The model failed to produce a prediction.") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    prediction_row = Prediction(
        request_id=request_id,
        model_version_id=model_version.id,
        input_features=features,
        prediction=labels[0],
        probability=probabilities[0],
        latency_ms=latency_ms,
    )
    db.add(prediction_row)
    db.commit()
    db.refresh(prediction_row)

    PREDICTIONS_TOTAL.labels(
        model_version=model_version.version_label, algorithm=model_version.algorithm
    ).inc()

    return PredictionResult(prediction_row=prediction_row, model_version=model_version)


def execute_batch_prediction(db: Session, job_id: uuid.UUID) -> None:
    """Run every record in a queued batch job and record the results.

    Called both inline (small/synchronous batches) and from the Celery
    worker (large/async batches) — this is the one place batch execution
    logic lives, so the two paths can't drift.

    The model version was resolved and pinned into job.payload when the
    job was created (see app/routers/predictions.py) — this function
    always uses that exact version, not "whatever is production now,"
    so a promotion that happens while a job is queued can't change which
    model a given batch is attributed to.
    """
    job = db.get(Job, job_id)
    if job is None:
        logger.error("batch_job_not_found", extra={"job_id": str(job_id)})
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC)
    db.commit()

    try:
        model_version_id = job.payload["model_version_id"]
        model_version = db.get(ModelVersionRecord, model_version_id)
        if model_version is None:
            raise PredictionExecutionError(
                f"Pinned model version {model_version_id} no longer exists."
            )

        pipeline = load_pipeline_for_model_version(model_version)
        records: list[dict] = job.payload["records"]
        df = _build_feature_dataframe(records)

        start = time.perf_counter()
        labels, probabilities = _labels_and_probabilities(pipeline, df)
        total_latency_ms = (time.perf_counter() - start) * 1000
        per_record_latency_ms = total_latency_ms / max(len(records), 1)

        prediction_ids = []
        for record, label, probability in zip(records, labels, probabilities, strict=True):
            prediction_row = Prediction(
                request_id=job.id,
                model_version_id=model_version.id,
                input_features=record,
                prediction=label,
                probability=probability,
                latency_ms=per_record_latency_ms,
            )
            db.add(prediction_row)
            db.flush()
            prediction_ids.append(str(prediction_row.id))

        PREDICTIONS_TOTAL.labels(
            model_version=model_version.version_label, algorithm=model_version.algorithm
        ).inc(len(records))

        job.status = JobStatus.COMPLETED
        job.result = {"n_processed": len(records), "prediction_ids": prediction_ids}
        job.finished_at = datetime.now(UTC)
        db.commit()

    except Exception as exc:
        db.rollback()
        PREDICTION_ERRORS_TOTAL.labels(reason="batch_execution_exception").inc()
        logger.exception("batch_prediction_failed", extra={"job_id": str(job_id)})
        job = db.get(Job, job_id)
        job.status = JobStatus.FAILED
        job.error_message = str(exc)[:2000]
        job.finished_at = datetime.now(UTC)
        db.commit()
