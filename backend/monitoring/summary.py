"""GET /monitoring/summary — a single real-state snapshot for a dashboard.

Every field is either a database query (predictions/jobs/model_versions)
or a live read of the existing Prometheus counters (via sum_counter_value,
the public collect() API — not a second counting system). Nothing here
is synthesized; an idle system correctly reports zeros.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.services.registry import get_production_model
from db.models import Job, JobStatus, JobType, Prediction
from monitoring.job_stats import get_training_job_stats
from monitoring.metrics import (
    BATCH_PREDICTIONS_TOTAL,
    PREDICTION_ERRORS_TOTAL,
    PREDICTIONS_TOTAL,
    sum_counter_value,
)
from monitoring.model_usage import get_model_version_usage

RECENT_WINDOW = timedelta(hours=24)


def get_monitoring_summary(db: Session) -> dict:
    production = get_production_model(db)
    production_summary = (
        {
            "model_version_id": str(production.id),
            "version_label": production.version_label,
            "algorithm": production.algorithm,
            "promoted_at": production.promoted_at.isoformat() if production.promoted_at else None,
        }
        if production is not None
        else None
    )

    cutoff = datetime.now(UTC) - RECENT_WINDOW
    recent_prediction_count = (
        db.query(Prediction).filter(Prediction.created_at >= cutoff).count()
    )

    active_statuses = [JobStatus.QUEUED, JobStatus.RUNNING]
    active_training_jobs = (
        db.query(Job)
        .filter(Job.job_type == JobType.TRAIN, Job.status.in_(active_statuses))
        .count()
    )

    recent_failed_jobs = (
        db.query(Job)
        .filter(Job.status == JobStatus.FAILED, Job.created_at >= cutoff)
        .order_by(Job.created_at.desc())
        .limit(5)
        .all()
    )

    drift_status = "no_production_model"
    if production is not None:
        from monitoring.drift import resolve_drift_report

        drift_status = resolve_drift_report(db, model_version_id=production.id).get("status")

    return {
        "current_production_model": production_summary,
        "recent_window_hours": RECENT_WINDOW.total_seconds() / 3600,
        "recent_prediction_count": recent_prediction_count,
        "recent_prediction_error_count": sum_counter_value(PREDICTION_ERRORS_TOTAL),
        "total_predictions_this_process": sum_counter_value(PREDICTIONS_TOTAL),
        "total_batch_requests_this_process": sum_counter_value(BATCH_PREDICTIONS_TOTAL),
        "drift_status": drift_status,
        "active_training_jobs": active_training_jobs,
        "recent_failed_jobs": [
            {
                "job_id": str(j.id),
                "job_type": j.job_type.value,
                "error_message": j.error_message,
                "created_at": j.created_at.isoformat(),
            }
            for j in recent_failed_jobs
        ],
        "training_job_stats": get_training_job_stats(db),
        "model_version_usage": get_model_version_usage(db),
    }
