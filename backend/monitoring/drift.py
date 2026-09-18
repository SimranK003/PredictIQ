"""Drift-detection foundation.

Phase 6 implements the actual reference-vs-current statistical
comparison (PSI/KS, per the original architecture plan). This module
only builds the piece that must exist first: a way to tell whether there
is *enough real production traffic* to say anything meaningful at all.

Below the configured threshold, this deliberately returns
"insufficient_data" rather than computing a distribution summary from a
handful of predictions and presenting it as if it meant something.
"""

from collections import Counter

from sqlalchemy.orm import Session

from app.core.config import get_settings
from db.models import Prediction
from ml.schema import CHURN_SCHEMA, DatasetSchema


def get_drift_status(
    db: Session, model_version_id, *, schema: DatasetSchema = CHURN_SCHEMA
) -> dict:
    """Real (not fake) foundation for drift monitoring, scoped to one
    model version — predictions from a superseded model shouldn't be
    blended into "current" traffic for whatever's in production now.
    """
    settings = get_settings()
    minimum = settings.drift_minimum_predictions

    n_predictions = (
        db.query(Prediction).filter(Prediction.model_version_id == model_version_id).count()
    )

    if n_predictions < minimum:
        return {
            "status": "insufficient_data",
            "model_version_id": str(model_version_id),
            "n_production_predictions": n_predictions,
            "minimum_required": minimum,
        }

    predictions = (
        db.query(Prediction)
        .filter(Prediction.model_version_id == model_version_id)
        .order_by(Prediction.created_at.desc())
        .limit(1000)
        .all()
    )

    numeric_summary = {}
    for col in schema.numeric_columns:
        values = [
            float(p.input_features[col])
            for p in predictions
            if p.input_features.get(col) is not None
        ]
        if values:
            numeric_summary[col] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

    categorical_summary = {}
    for col in schema.categorical_columns:
        values = [
            p.input_features.get(col)
            for p in predictions
            if p.input_features.get(col) is not None
        ]
        counts = Counter(values)
        categorical_summary[col] = dict(counts.most_common())

    return {
        "status": "summary_available",
        "model_version_id": str(model_version_id),
        "n_production_predictions": n_predictions,
        "n_sampled": len(predictions),
        "numeric_feature_summary": numeric_summary,
        "categorical_feature_distribution": categorical_summary,
        "note": (
            "Descriptive statistics only — this is not a drift score. "
            "Reference-vs-current statistical comparison (PSI/KS) is Phase 6."
        ),
    }


def resolve_production_drift_status(db: Session) -> dict:
    from app.services.registry import get_production_model

    production = get_production_model(db)
    if production is None:
        return {"status": "no_production_model"}
    return get_drift_status(db, production.id)
