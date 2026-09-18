"""Tests for the production model cache (app/services/model_loader.py).

This is what a future /predict endpoint (Phase 4) will call — these
tests are the current proof that "inference resolves the model marked
production" and "does not reload from disk/MLflow on every request."
"""

import pytest

from app.core.exceptions import NotFoundError
from app.services import model_loader
from db.models import ModelStage


@pytest.fixture(autouse=True)
def _reset_cache():
    model_loader.invalidate_production_model_cache()
    yield
    model_loader.invalidate_production_model_cache()


def test_raises_not_found_when_no_production_model(db_session):
    with pytest.raises(NotFoundError):
        model_loader.get_production_pipeline(db_session)


def test_resolves_the_model_marked_production(db_session, make_model_version, monkeypatch):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    make_model_version(stage=ModelStage.CANDIDATE)  # must NOT be resolved

    monkeypatch.setattr(
        model_loader.mlflow.sklearn, "load_model", lambda uri: f"pipeline-for:{uri}"
    )

    resolved_version, pipeline = model_loader.get_production_pipeline(db_session)

    assert resolved_version.id == production.id
    assert pipeline == f"pipeline-for:{production.artifact_uri}"


def test_does_not_reload_when_production_is_unchanged(db_session, make_model_version, monkeypatch):
    make_model_version(stage=ModelStage.PRODUCTION)
    load_calls = []
    monkeypatch.setattr(
        model_loader.mlflow.sklearn,
        "load_model",
        lambda uri: load_calls.append(uri) or "loaded-pipeline",
    )

    model_loader.get_production_pipeline(db_session)
    model_loader.get_production_pipeline(db_session)
    model_loader.get_production_pipeline(db_session)

    assert len(load_calls) == 1, "expected the pipeline to be loaded once and reused from cache"


def test_reloads_when_production_model_changes(db_session, make_model_version, monkeypatch):
    first = make_model_version(stage=ModelStage.PRODUCTION)
    load_calls = []
    monkeypatch.setattr(
        model_loader.mlflow.sklearn,
        "load_model",
        lambda uri: load_calls.append(uri) or f"pipeline:{uri}",
    )

    resolved_first, _ = model_loader.get_production_pipeline(db_session)
    assert resolved_first.id == first.id
    assert len(load_calls) == 1

    # Simulate a promotion: first goes to previous, a new one becomes
    # production. Flush the demotion before inserting the new production
    # row — the partial unique index checks per-statement, and SQLAlchemy
    # doesn't guarantee this UPDATE flushes before that INSERT otherwise.
    first.stage = ModelStage.PREVIOUS
    db_session.flush()
    second = make_model_version(stage=ModelStage.PRODUCTION)

    resolved_second, pipeline = model_loader.get_production_pipeline(db_session)
    assert resolved_second.id == second.id
    assert len(load_calls) == 2, "expected a reload after the production model changed"
    assert pipeline == f"pipeline:{second.artifact_uri}"


def test_invalidate_forces_a_reload_even_if_production_id_unchanged(
    db_session, make_model_version, monkeypatch
):
    make_model_version(stage=ModelStage.PRODUCTION)
    load_calls = []
    monkeypatch.setattr(
        model_loader.mlflow.sklearn,
        "load_model",
        lambda uri: load_calls.append(uri) or "pipeline",
    )

    model_loader.get_production_pipeline(db_session)
    model_loader.invalidate_production_model_cache()
    model_loader.get_production_pipeline(db_session)

    assert len(load_calls) == 2
