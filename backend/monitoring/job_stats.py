"""Training job statistics, computed from the real jobs table only —
no synthetic values, no separate job-metrics store.
"""

from sqlalchemy.orm import Session

from db.models import Job, JobStatus, JobType


def get_training_job_stats(db: Session, *, recent_failures_limit: int = 5) -> dict:
    jobs = db.query(Job).filter(Job.job_type == JobType.TRAIN).all()

    total = len(jobs)
    completed = [j for j in jobs if j.status == JobStatus.COMPLETED]
    failed = [j for j in jobs if j.status == JobStatus.FAILED]
    active = [j for j in jobs if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)]

    durations = [
        (j.finished_at - j.started_at).total_seconds()
        for j in completed
        if j.started_at is not None and j.finished_at is not None
    ]
    average_duration_seconds = round(sum(durations) / len(durations), 2) if durations else None

    recent_failures = sorted(failed, key=lambda j: j.created_at, reverse=True)
    recent_failures = recent_failures[:recent_failures_limit]

    return {
        "total_jobs": total,
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "active_jobs": len(active),
        "average_training_duration_seconds": average_duration_seconds,
        "n_durations_sampled": len(durations),
        "recent_failures": [
            {
                "job_id": str(j.id),
                "dataset_id": str(j.dataset_id) if j.dataset_id else None,
                "error_message": j.error_message,
                "created_at": j.created_at.isoformat(),
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            }
            for j in recent_failures
        ],
    }
