"""Real statistical drift detection.

Methodology (full writeup in docs/monitoring.md):

- Numeric features -> two-sample Kolmogorov-Smirnov test
  (scipy.stats.ks_2samp). KS compares two empirical CDFs directly and
  needs no binning choice, which makes it the natural fit for continuous
  values like tenure/MonthlyCharges/TotalCharges — an arbitrary bin
  width would otherwise change the answer. The KS statistic (0-1) is
  the maximum vertical gap between the two CDFs; drift_detected compares
  it against DRIFT_KS_THRESHOLD.

- Categorical features -> Population Stability Index (PSI) over category
  proportions. KS assumes an ordered, continuous variable — applying it
  to an unordered category like PaymentMethod is not meaningful (there
  is no "CDF" of payment methods). PSI instead compares the proportion
  in each category directly, which is exactly the right shape for
  categorical data, and is the standard choice in the ML-ops/credit-risk
  literature for this reason. drift_detected compares PSI against
  DRIFT_PSI_WARNING; a DRIFT_PSI_CRITICAL band is also reported for
  extra nuance the base status doesn't distinguish.

Reference distribution: for each model version, the reference is built
by *re-reading the exact training dataset that model was trained on*
(model_version.dataset_id -> Dataset.storage_path), never a separate or
"current" dataset — see build_reference_distribution(). This is what
keeps a drift comparison honest when the production model has changed:
comparing today's traffic against the wrong model's training data would
either hide real drift or invent fake drift.

Below DRIFT_MIN_SAMPLES real production predictions for the requested
model version (and time window, if any), this returns "insufficient_data"
— never a score computed from too few points to mean anything.
"""

import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from db.models import Dataset, ModelVersionRecord, Prediction
from ml.schema import CHURN_SCHEMA, DatasetSchema
from ml.training.data import load_raw_dataframe

# Categories seen in production but never in training (or vice versa)
# get this small nonzero proportion instead of 0, so PSI's log() term
# doesn't blow up to +/-inf on a single novel value. This is a standard
# PSI smoothing convention, not a project-specific tuning.
_PSI_EPSILON = 1e-4

WINDOW_PRESETS = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}


@dataclass
class ReferenceDistribution:
    model_version_id: uuid.UUID
    dataset_id: uuid.UUID
    dataset_content_hash: str
    training_job_id: uuid.UUID | None
    numeric_values: dict[str, np.ndarray]
    categorical_proportions: dict[str, dict[str, float]]
    n_rows: int


def build_reference_distribution(
    db: Session, model_version: ModelVersionRecord, *, schema: DatasetSchema = CHURN_SCHEMA
) -> ReferenceDistribution:
    """The reference is the model's own training dataset — traceable via
    model_version -> dataset_id -> Dataset.storage_path/content_hash, and
    (when available) model_version -> training_job_id -> Job.dataset_id
    as an independent cross-check of the same lineage. Never a different
    or "global" dataset.
    """
    dataset = db.get(Dataset, model_version.dataset_id)
    if dataset is None:
        raise NotFoundError(
            f"Training dataset {model_version.dataset_id} for model version "
            f"{model_version.id} no longer exists — cannot build a reference distribution."
        )

    df = load_raw_dataframe(dataset.storage_path)

    numeric_values = {}
    for col in schema.numeric_columns:
        numeric_values[col] = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

    categorical_proportions = {}
    for col in schema.categorical_columns:
        categorical_proportions[col] = df[col].value_counts(normalize=True).to_dict()

    return ReferenceDistribution(
        model_version_id=model_version.id,
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        training_job_id=model_version.training_job_id,
        numeric_values=numeric_values,
        categorical_proportions=categorical_proportions,
        n_rows=len(df),
    )


def compute_ks_drift(
    reference_values: np.ndarray, production_values: np.ndarray
) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    result = stats.ks_2samp(reference_values, production_values)
    return float(result.statistic), float(result.pvalue)


def compute_psi_categorical(
    reference_proportions: dict[str, float], production_values: list[str]
) -> float:
    """PSI = sum_over_categories[(prod% - ref%) * ln(prod% / ref%)].

    Categories present in only one side are handled by treating their
    missing-side proportion as _PSI_EPSILON rather than 0, which would
    otherwise make ln() undefined (division by zero / log of zero) —
    this is the standard PSI smoothing convention: an unseen category
    isn't "impossible," just rare enough not to have appeared yet.
    """
    if not production_values:
        return 0.0

    prod_counts = Counter(production_values)
    total_prod = len(production_values)
    categories = set(reference_proportions) | set(prod_counts)

    psi = 0.0
    for category in categories:
        ref_p = reference_proportions.get(category, 0.0) or _PSI_EPSILON
        prod_p = (prod_counts.get(category, 0) / total_prod) or _PSI_EPSILON
        psi += (prod_p - ref_p) * np.log(prod_p / ref_p)

    return float(psi)


def _numeric_feature_result(
    feature: str, reference_values: np.ndarray, production_values: np.ndarray, ks_threshold: float
) -> dict:
    if len(reference_values) == 0 or len(production_values) == 0:
        return {
            "feature": feature,
            "test": "KS",
            "score": None,
            "threshold": ks_threshold,
            "drift_detected": False,
            "note": "no usable values for this feature on one side of the comparison",
        }

    statistic, p_value = compute_ks_drift(reference_values, production_values)
    return {
        "feature": feature,
        "test": "KS",
        "score": round(statistic, 4),
        "p_value": round(p_value, 4),
        "threshold": ks_threshold,
        "drift_detected": statistic >= ks_threshold,
        "reference_n": len(reference_values),
        "production_n": len(production_values),
    }


def _categorical_feature_result(
    feature: str,
    reference_proportions: dict[str, float],
    production_values: list[str],
    *,
    psi_warning: float,
    psi_critical: float,
) -> dict:
    if not production_values or not reference_proportions:
        return {
            "feature": feature,
            "test": "PSI",
            "score": None,
            "threshold": psi_warning,
            "drift_detected": False,
            "note": "no usable values for this feature on one side of the comparison",
        }

    psi = compute_psi_categorical(reference_proportions, production_values)
    severity = "critical" if psi >= psi_critical else "warning" if psi >= psi_warning else "none"
    return {
        "feature": feature,
        "test": "PSI",
        "score": round(psi, 4),
        "threshold": psi_warning,
        "critical_threshold": psi_critical,
        "severity": severity,
        "drift_detected": psi >= psi_warning,
        "production_n": len(production_values),
    }


def compute_drift_report(
    db: Session,
    model_version_id,
    *,
    window: str | None = None,
    feature: str | None = None,
    schema: DatasetSchema = CHURN_SCHEMA,
) -> dict:
    """The real drift computation. Raises NotFoundError for an unknown
    model_version_id and ValueError for an unknown feature name or
    window preset — the router maps these to 404/400.
    """
    settings = get_settings()

    model_version = db.get(ModelVersionRecord, model_version_id)
    if model_version is None:
        raise NotFoundError(f"Model version {model_version_id} not found.")

    if window is not None and window not in WINDOW_PRESETS:
        raise ValueError(f"Unknown window {window!r}. Valid: {sorted(WINDOW_PRESETS)}")

    features_to_check = [feature] if feature else list(schema.all_feature_columns)
    unknown_features = set(features_to_check) - set(schema.all_feature_columns)
    if unknown_features:
        raise ValueError(
            f"Unknown feature(s): {sorted(unknown_features)}. "
            f"Valid: {list(schema.all_feature_columns)}"
        )

    query = db.query(Prediction).filter(Prediction.model_version_id == model_version_id)
    if window is not None:
        cutoff = datetime.now(UTC) - WINDOW_PRESETS[window]
        query = query.filter(Prediction.created_at >= cutoff)

    sample_size = query.count()

    base_response = {
        "model_version": model_version.version_label,
        "model_version_id": str(model_version.id),
        "window": window,
        "sample_size": sample_size,
        "minimum_required": settings.drift_min_samples,
    }

    if sample_size < settings.drift_min_samples:
        return {**base_response, "status": "insufficient_data"}

    predictions = (
        query.order_by(Prediction.created_at.desc())
        .limit(settings.drift_max_predictions_sampled)
        .all()
    )
    reference = build_reference_distribution(db, model_version, schema=schema)

    results = []
    for col in features_to_check:
        if col in schema.numeric_columns:
            production_values = np.array(
                [
                    float(p.input_features[col])
                    for p in predictions
                    if p.input_features.get(col) not in (None, "")
                ]
            )
            results.append(
                _numeric_feature_result(
                    col,
                    reference.numeric_values[col],
                    production_values,
                    settings.drift_ks_threshold,
                )
            )
        else:
            production_values = [
                p.input_features.get(col)
                for p in predictions
                if p.input_features.get(col) is not None
            ]
            results.append(
                _categorical_feature_result(
                    col,
                    reference.categorical_proportions.get(col, {}),
                    production_values,
                    psi_warning=settings.drift_psi_warning,
                    psi_critical=settings.drift_psi_critical,
                )
            )

    return {
        **base_response,
        "status": "ok",
        "n_sampled_for_computation": len(predictions),
        "reference_dataset_id": str(reference.dataset_id),
        "reference_dataset_content_hash": reference.dataset_content_hash,
        "reference_training_job_id": (
            str(reference.training_job_id) if reference.training_job_id else None
        ),
        "reference_n_rows": reference.n_rows,
        "drift_detected": any(r["drift_detected"] for r in results),
        "features": results,
    }


def resolve_drift_report(
    db: Session,
    *,
    model_version_id=None,
    window: str | None = None,
    feature: str | None = None,
) -> dict:
    """Defaults to the current production model when no model_version_id
    is given — the common case for a dashboard's default view.
    """
    if model_version_id is None:
        from app.services.registry import get_production_model

        production = get_production_model(db)
        if production is None:
            return {"status": "no_production_model"}
        model_version_id = production.id

    return compute_drift_report(db, model_version_id, window=window, feature=feature)
