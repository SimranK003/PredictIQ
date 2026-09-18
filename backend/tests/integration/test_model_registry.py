"""Registry lifecycle tests: register -> promote -> demote -> rollback.

Runs against real Postgres (via the db_session fixture) since this is
exactly the kind of logic — transactions, stage transitions — that a
mocked DB would validate the wrong thing for.
"""

import pytest

from app.core.exceptions import InvalidPromotionError, NotFoundError
from app.services.registry import (
    get_production_model,
    promote_model,
    register_candidate,
    rollback_model,
)
from db.models import LifecycleAction, ModelStage


def _always_true(_run_id: str) -> bool:
    return True


def _always_false(_run_id: str) -> bool:
    return False


# --- Registration -----------------------------------------------------


def test_register_candidate_creates_candidate_with_dataset_lineage(db_session, make_dataset):
    dataset = make_dataset()
    candidate = register_candidate(
        db_session,
        dataset=dataset,
        algorithm="logistic_regression",
        mlflow_run_id="run-abc",
        mlflow_experiment_id="1",
        artifact_uri="runs:/run-abc/model",
        metrics={"precision": 0.5, "recall": 0.6, "f1": 0.55, "roc_auc": 0.8},
        params={"C": 1.0},
        git_commit="deadbeef",
    )

    assert candidate.stage == ModelStage.CANDIDATE
    assert candidate.dataset_id == dataset.id
    assert candidate.mlflow_run_id == "run-abc"
    assert candidate.version_label.startswith("v")


def test_register_candidate_assigns_sequential_version_labels(db_session, make_dataset):
    dataset = make_dataset()
    first = register_candidate(
        db_session,
        dataset=dataset,
        algorithm="logistic_regression",
        mlflow_run_id="run-1",
        mlflow_experiment_id="1",
        artifact_uri="runs:/run-1/model",
        metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5},
        params={},
        git_commit="c1",
    )
    second = register_candidate(
        db_session,
        dataset=dataset,
        algorithm="random_forest",
        mlflow_run_id="run-2",
        mlflow_experiment_id="1",
        artifact_uri="runs:/run-2/model",
        metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5},
        params={},
        git_commit="c2",
    )
    assert first.version_label != second.version_label


def test_register_candidate_records_lifecycle_event(db_session, make_dataset):
    dataset = make_dataset()
    candidate = register_candidate(
        db_session,
        dataset=dataset,
        algorithm="xgboost",
        mlflow_run_id="run-xyz",
        mlflow_experiment_id="1",
        artifact_uri="runs:/run-xyz/model",
        metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5},
        params={},
        git_commit="c3",
    )
    db_session.refresh(candidate)
    events = candidate.lifecycle_events
    assert len(events) == 1
    assert events[0].action == LifecycleAction.REGISTERED
    assert events[0].new_stage == ModelStage.CANDIDATE
    assert events[0].triggered_by == "system"


# --- Promotion: candidate -> production --------------------------------


def test_promote_candidate_with_no_existing_production_succeeds(db_session, make_model_version):
    candidate = make_model_version(stage=ModelStage.CANDIDATE)

    promoted = promote_model(db_session, candidate.id, artifact_checker=_always_true)

    assert promoted.stage == ModelStage.PRODUCTION
    assert promoted.promoted_at is not None
    assert get_production_model(db_session).id == candidate.id


def test_promote_demotes_existing_production_to_previous(db_session, make_model_version):
    old_production = make_model_version(stage=ModelStage.PRODUCTION)
    new_candidate = make_model_version(stage=ModelStage.CANDIDATE)

    promote_model(db_session, new_candidate.id, artifact_checker=_always_true)

    db_session.refresh(old_production)
    db_session.refresh(new_candidate)
    assert old_production.stage == ModelStage.PREVIOUS
    assert new_candidate.stage == ModelStage.PRODUCTION


def test_promote_archives_existing_previous_when_demoting_new_previous(
    db_session, make_model_version
):
    """Production v1, promote v2 (v1->previous), promote v3: v1 should be
    archived (only one "previous" slot exists) and v2 becomes previous.
    """
    v1 = make_model_version(stage=ModelStage.PRODUCTION)
    v2 = make_model_version(stage=ModelStage.CANDIDATE)
    promote_model(db_session, v2.id, artifact_checker=_always_true)

    v3 = make_model_version(stage=ModelStage.CANDIDATE)
    promote_model(db_session, v3.id, artifact_checker=_always_true)

    db_session.refresh(v1)
    db_session.refresh(v2)
    db_session.refresh(v3)
    assert v1.stage == ModelStage.ARCHIVED
    assert v2.stage == ModelStage.PREVIOUS
    assert v3.stage == ModelStage.PRODUCTION


def test_promote_records_lifecycle_events_for_all_affected_versions(db_session, make_model_version):
    old_production = make_model_version(stage=ModelStage.PRODUCTION)
    candidate = make_model_version(stage=ModelStage.CANDIDATE)

    promote_model(db_session, candidate.id, artifact_checker=_always_true)

    db_session.refresh(old_production)
    db_session.refresh(candidate)
    old_events = old_production.lifecycle_events
    assert any(e.action == LifecycleAction.DEMOTED_TO_PREVIOUS for e in old_events)
    assert any(e.action == LifecycleAction.PROMOTED for e in candidate.lifecycle_events)


# --- Promotion safety policy --------------------------------------------


def test_promote_rejects_non_candidate_stage(db_session, make_model_version):
    already_production = make_model_version(stage=ModelStage.PRODUCTION)

    with pytest.raises(InvalidPromotionError, match="not a candidate"):
        promote_model(db_session, already_production.id, artifact_checker=_always_true)


def test_promote_rejects_nonexistent_model(db_session):
    with pytest.raises(NotFoundError):
        fake_id = "00000000-0000-0000-0000-000000000000"
        promote_model(db_session, fake_id, artifact_checker=_always_true)


def test_promote_rejects_missing_required_metrics(db_session, make_model_version):
    candidate = make_model_version(
        stage=ModelStage.CANDIDATE, metrics={"precision": 0.5, "recall": 0.5}
    )
    with pytest.raises(InvalidPromotionError, match="missing required evaluation metric"):
        promote_model(db_session, candidate.id, artifact_checker=_always_true)


def test_promote_rejects_candidate_with_invalid_dataset(
    db_session, make_dataset, make_model_version
):
    bad_dataset = make_dataset(is_valid=False)
    candidate = make_model_version(stage=ModelStage.CANDIDATE, dataset=bad_dataset)

    with pytest.raises(InvalidPromotionError, match="dataset"):
        promote_model(db_session, candidate.id, artifact_checker=_always_true)


def test_promote_rejects_when_artifact_cannot_be_verified(db_session, make_model_version):
    candidate = make_model_version(stage=ModelStage.CANDIDATE)

    with pytest.raises(InvalidPromotionError, match="artifact"):
        promote_model(db_session, candidate.id, artifact_checker=_always_false)


def test_promote_does_not_mutate_state_when_policy_check_fails(db_session, make_model_version):
    candidate = make_model_version(
        stage=ModelStage.CANDIDATE, metrics={"precision": 0.5, "recall": 0.5}
    )
    with pytest.raises(InvalidPromotionError):
        promote_model(db_session, candidate.id, artifact_checker=_always_true)

    db_session.refresh(candidate)
    assert candidate.stage == ModelStage.CANDIDATE
    assert candidate.promoted_at is None


# --- Rollback ------------------------------------------------------------


def test_rollback_swaps_production_and_previous(db_session, make_model_version):
    production = make_model_version(stage=ModelStage.PRODUCTION, version_label="v3")
    previous = make_model_version(stage=ModelStage.PREVIOUS, version_label="v2")

    restored = rollback_model(db_session)

    db_session.refresh(production)
    db_session.refresh(previous)
    assert restored.id == previous.id
    assert previous.stage == ModelStage.PRODUCTION
    assert production.stage == ModelStage.PREVIOUS


def test_rollback_raises_when_no_previous_model_exists(db_session, make_model_version):
    make_model_version(stage=ModelStage.PRODUCTION)
    with pytest.raises(InvalidPromotionError, match="No previous"):
        rollback_model(db_session)


def test_rollback_records_lifecycle_events(db_session, make_model_version):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    previous = make_model_version(stage=ModelStage.PREVIOUS)

    rollback_model(db_session)

    db_session.refresh(production)
    db_session.refresh(previous)
    assert any(e.action == LifecycleAction.ROLLED_BACK for e in production.lifecycle_events)
    assert any(e.action == LifecycleAction.ROLLED_BACK for e in previous.lifecycle_events)


# --- Read helpers ---------------------------------------------------------


def test_get_production_model_returns_none_when_none_registered(db_session):
    assert get_production_model(db_session) is None


def test_get_production_model_returns_current_production(db_session, make_model_version):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    make_model_version(stage=ModelStage.CANDIDATE)  # noise

    result = get_production_model(db_session)
    assert result is not None
    assert result.id == production.id
