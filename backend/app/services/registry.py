"""Model registry: candidate registration, promotion, and rollback.

Architecture: MLflow stays responsible for experiment tracking (params,
metrics, artifacts) — nothing here re-implements that. This module is
the *application-level* lifecycle on top of it: it snapshots just enough
from a finished MLflow run (algorithm, final metrics, artifact URI) into
Postgres so the promotion workflow, dashboard, and inference layer never
need to call out to MLflow to answer "what's in production and how good
is it." Full run history, hyperparameter search, and artifacts remain
queryable from MLflow directly via mlflow_run_id.

Promotion safety: candidates are never auto-promoted. POST /models/promote
is the only path to production, and it's gated by an explicit, configurable
policy (see _check_promotion_policy). The single-production and
single-previous invariants are enforced by Postgres partial unique indexes
(db/models.py), not just this code — a race between two concurrent
promotions is caught at commit time and surfaced as PromotionConflictError,
not silently resolved by whichever request happened to run last.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import mlflow
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import InvalidPromotionError, NotFoundError, PromotionConflictError
from db.models import (
    Dataset,
    LifecycleAction,
    ModelLifecycleEvent,
    ModelStage,
    ModelVersionRecord,
)


def default_artifact_exists(mlflow_run_id: str, artifact_path: str = "model") -> bool:
    """Live check that the run's model artifact is actually present.

    Fails safe: any error (missing run, unreachable tracking server,
    missing artifact) returns False, which blocks promotion rather than
    silently proceeding.
    """
    try:
        artifacts = mlflow.artifacts.list_artifacts(
            run_id=mlflow_run_id, artifact_path=artifact_path
        )
        return len(artifacts) > 0
    except Exception:
        return False


def _next_version_label(db: Session) -> str:
    count = db.query(ModelVersionRecord).count()
    return f"v{count + 1}"


def _record_event(
    db: Session,
    model_version: ModelVersionRecord,
    *,
    action: LifecycleAction,
    previous_stage: ModelStage | None,
    new_stage: ModelStage,
    triggered_by: str,
    metadata: dict | None = None,
) -> None:
    db.add(
        ModelLifecycleEvent(
            model_version_id=model_version.id,
            action=action,
            previous_stage=previous_stage,
            new_stage=new_stage,
            triggered_by=triggered_by,
            event_metadata=metadata or {},
        )
    )


def register_candidate(
    db: Session,
    *,
    dataset: Dataset,
    algorithm: str,
    mlflow_run_id: str,
    mlflow_experiment_id: str,
    artifact_uri: str,
    metrics: dict,
    params: dict,
    git_commit: str,
    training_job_id=None,
) -> ModelVersionRecord:
    """Record a finished MLflow run as a new candidate model version.

    This is called automatically at the end of training (ml/training/train.py,
    directly or via the async training Celery task) — registering a
    *candidate* is not the same as promoting it, so it's safe to
    automate: nothing in this function makes the model servable.

    training_job_id is nullable: candidates registered by the original
    CLI training path (Phase 2, before async training jobs existed) have
    no Job row, which is an honest gap rather than something to backfill.
    """
    model_version = ModelVersionRecord(
        version_label=_next_version_label(db),
        algorithm=algorithm,
        mlflow_run_id=mlflow_run_id,
        mlflow_experiment_id=mlflow_experiment_id,
        artifact_uri=artifact_uri,
        git_commit=git_commit,
        stage=ModelStage.CANDIDATE,
        metrics=metrics,
        params=params,
        dataset_id=dataset.id,
        training_job_id=training_job_id,
    )
    db.add(model_version)
    db.flush()  # assign model_version.id for the event FK

    _record_event(
        db,
        model_version,
        action=LifecycleAction.REGISTERED,
        previous_stage=None,
        new_stage=ModelStage.CANDIDATE,
        triggered_by="system",
    )
    db.commit()
    db.refresh(model_version)
    return model_version


def _check_promotion_policy(
    db: Session,
    candidate: ModelVersionRecord,
    *,
    artifact_checker: Callable[[str], bool],
) -> None:
    settings = get_settings()

    if candidate.stage != ModelStage.CANDIDATE:
        raise InvalidPromotionError(
            f"Model version {candidate.id} is not a candidate (current stage: "
            f"{candidate.stage.value}); only candidates can be promoted."
        )

    if candidate.dataset_id is None:
        raise InvalidPromotionError("Candidate has no linked training dataset.")

    dataset = db.get(Dataset, candidate.dataset_id)
    if dataset is None or not dataset.is_valid:
        raise InvalidPromotionError(
            "Candidate's training dataset is missing or failed validation."
        )

    required_metrics = settings.model_promotion_required_metrics_list
    missing = [m for m in required_metrics if candidate.metrics.get(m) is None]
    if missing:
        raise InvalidPromotionError(
            f"Candidate is missing required evaluation metric(s): {missing}."
        )

    if settings.model_promotion_require_artifact_check:
        if not artifact_checker(candidate.mlflow_run_id):
            raise InvalidPromotionError(
                f"Candidate's MLflow artifact could not be verified for run "
                f"{candidate.mlflow_run_id}; refusing to promote an unverifiable model."
            )


def promote_model(
    db: Session,
    candidate_id,
    *,
    triggered_by: str = "api",
    artifact_checker: Callable[[str], bool] = default_artifact_exists,
) -> ModelVersionRecord:
    """Promote a candidate to production.

    current production -> previous (if one exists)
    current previous    -> archived (if one exists; only one "previous" fits)
    candidate            -> production

    All in one transaction. The partial unique indexes on `stage` are the
    real safety net: even if two requests both pass the checks above
    concurrently, only one commit can succeed — the loser gets
    PromotionConflictError, not a silently-corrupted two-production state.

    Postgres checks a unique index per-statement, and a partial unique
    index can't be made DEFERRABLE (Postgres only allows deferring real
    UNIQUE/PK constraints, which can't carry a WHERE clause) — so the
    three stage changes below are flushed in an order that never puts two
    rows in the same exclusive stage at the same time, rather than
    relying on all-or-nothing visibility at commit.
    """
    candidate = db.execute(
        select(ModelVersionRecord).where(ModelVersionRecord.id == candidate_id).with_for_update()
    ).scalar_one_or_none()
    if candidate is None:
        raise NotFoundError(f"Model version {candidate_id} not found.")

    _check_promotion_policy(db, candidate, artifact_checker=artifact_checker)

    current_production = db.execute(
        select(ModelVersionRecord)
        .where(ModelVersionRecord.stage == ModelStage.PRODUCTION)
        .with_for_update()
    ).scalar_one_or_none()
    current_previous = db.execute(
        select(ModelVersionRecord)
        .where(ModelVersionRecord.stage == ModelStage.PREVIOUS)
        .with_for_update()
    ).scalar_one_or_none()

    # 1. Free the "previous" slot first.
    if current_previous is not None:
        current_previous.stage = ModelStage.ARCHIVED
        _record_event(
            db,
            current_previous,
            action=LifecycleAction.ARCHIVED,
            previous_stage=ModelStage.PREVIOUS,
            new_stage=ModelStage.ARCHIVED,
            triggered_by=triggered_by,
            metadata={"reason": f"superseded by promotion of {candidate.id}"},
        )
        db.flush()

    # 2. Now safe to move current production into the now-empty "previous"
    #    slot — this also frees the "production" slot.
    if current_production is not None:
        current_production.stage = ModelStage.PREVIOUS
        _record_event(
            db,
            current_production,
            action=LifecycleAction.DEMOTED_TO_PREVIOUS,
            previous_stage=ModelStage.PRODUCTION,
            new_stage=ModelStage.PREVIOUS,
            triggered_by=triggered_by,
            metadata={"reason": f"superseded by promotion of {candidate.id}"},
        )
        db.flush()

    # 3. Now safe to move the candidate into the now-empty "production" slot.
    candidate.stage = ModelStage.PRODUCTION
    candidate.promoted_at = datetime.now(UTC)
    _record_event(
        db,
        candidate,
        action=LifecycleAction.PROMOTED,
        previous_stage=ModelStage.CANDIDATE,
        new_stage=ModelStage.PRODUCTION,
        triggered_by=triggered_by,
    )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise PromotionConflictError(
            "A concurrent promotion committed first; this model version was not promoted. "
            "Refresh and retry if it's still a candidate."
        ) from exc

    db.refresh(candidate)
    return candidate


def rollback_model(db: Session, *, triggered_by: str = "api") -> ModelVersionRecord:
    """Swap production and previous: production -> previous, previous -> production.

    Raises InvalidPromotionError if there is no previous production model
    to roll back to. See promote_model's docstring for why the stage
    changes are flushed one at a time in a specific order rather than
    just assigned and committed together — a partial unique index can't
    be deferred, so an intermediate two-rows-in-the-same-stage state
    would be rejected even though the final state is valid.
    """
    current_production = db.execute(
        select(ModelVersionRecord)
        .where(ModelVersionRecord.stage == ModelStage.PRODUCTION)
        .with_for_update()
    ).scalar_one_or_none()
    current_previous = db.execute(
        select(ModelVersionRecord)
        .where(ModelVersionRecord.stage == ModelStage.PREVIOUS)
        .with_for_update()
    ).scalar_one_or_none()

    if current_previous is None:
        raise InvalidPromotionError("No previous production model is available to roll back to.")

    # 1. Move current_previous out of the "previous" slot temporarily so
    #    step 2 can freely place current_production there.
    current_previous.stage = ModelStage.ARCHIVED
    db.flush()

    # 2. Now safe: current production -> previous (frees the "production" slot).
    if current_production is not None:
        current_production.stage = ModelStage.PREVIOUS
        _record_event(
            db,
            current_production,
            action=LifecycleAction.ROLLED_BACK,
            previous_stage=ModelStage.PRODUCTION,
            new_stage=ModelStage.PREVIOUS,
            triggered_by=triggered_by,
            metadata={"reason": f"rolled back in favor of {current_previous.id}"},
        )
        db.flush()

    # 3. Now safe: former previous -> production.
    current_previous.stage = ModelStage.PRODUCTION
    current_previous.promoted_at = datetime.now(UTC)
    _record_event(
        db,
        current_previous,
        action=LifecycleAction.ROLLED_BACK,
        previous_stage=ModelStage.PREVIOUS,
        new_stage=ModelStage.PRODUCTION,
        triggered_by=triggered_by,
    )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise PromotionConflictError(
            "A concurrent promotion/rollback committed first; retry against current state."
        ) from exc

    db.refresh(current_previous)
    return current_previous


def get_production_model(db: Session) -> ModelVersionRecord | None:
    return db.execute(
        select(ModelVersionRecord).where(ModelVersionRecord.stage == ModelStage.PRODUCTION)
    ).scalar_one_or_none()
