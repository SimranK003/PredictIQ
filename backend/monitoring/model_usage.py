"""Model-version usage: which models are actually serving traffic.

Answers "is an old model still serving predictions after a promotion?" —
a real operational question, computed from the predictions table's exact
model_version_id (never the mutable "production" label), grouped by
version, with first/last-seen timestamps so a stale model still handling
traffic is visible at a glance.
"""

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import ModelVersionRecord, Prediction


def get_model_version_usage(db: Session) -> list[dict]:
    rows = (
        db.query(
            Prediction.model_version_id,
            func.count(Prediction.id).label("prediction_count"),
            func.min(Prediction.created_at).label("first_prediction_at"),
            func.max(Prediction.created_at).label("last_prediction_at"),
        )
        .group_by(Prediction.model_version_id)
        .all()
    )

    total_predictions = sum(row.prediction_count for row in rows)

    usage = []
    for row in rows:
        model_version = db.get(ModelVersionRecord, row.model_version_id)
        usage.append(
            {
                "model_version_id": str(row.model_version_id),
                "version_label": model_version.version_label if model_version else None,
                "algorithm": model_version.algorithm if model_version else None,
                "stage": model_version.stage.value if model_version else None,
                "prediction_count": row.prediction_count,
                "percentage_of_total": (
                    round(100 * row.prediction_count / total_predictions, 2)
                    if total_predictions
                    else 0.0
                ),
                "first_prediction_at": row.first_prediction_at.isoformat(),
                "last_prediction_at": row.last_prediction_at.isoformat(),
            }
        )

    usage.sort(key=lambda u: u["prediction_count"], reverse=True)
    return usage
