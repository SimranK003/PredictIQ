"""API tests for the model registry endpoints."""

import pytest

from app.core.config import get_settings
from db.models import ModelStage


@pytest.fixture()
def disable_artifact_check(monkeypatch):
    """Exercise the real promote_model end-to-end through the API without
    a live MLflow server, using the actual documented policy toggle
    (MODEL_PROMOTION_REQUIRE_ARTIFACT_CHECK) rather than stubbing the
    registry function itself.
    """
    monkeypatch.setenv("MODEL_PROMOTION_REQUIRE_ARTIFACT_CHECK", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_list_models_empty(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_models_returns_registered_versions(client, make_model_version):
    make_model_version(stage=ModelStage.CANDIDATE, version_label="v1")
    make_model_version(stage=ModelStage.PRODUCTION, version_label="v2")

    resp = client.get("/models")
    assert resp.status_code == 200
    labels = {m["version_label"] for m in resp.json()}
    assert labels == {"v1", "v2"}


def test_list_models_filters_by_stage(client, make_model_version):
    make_model_version(stage=ModelStage.CANDIDATE)
    production = make_model_version(stage=ModelStage.PRODUCTION)

    resp = client.get("/models", params={"stage": "production"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == str(production.id)


def test_get_model_by_id(client, make_model_version):
    model_version = make_model_version()
    resp = client.get(f"/models/{model_version.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(model_version.id)
    assert resp.json()["git_commit"] == model_version.git_commit


def test_get_model_by_id_returns_404_when_missing(client):
    resp = client.get("/models/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_get_model_metrics(client, make_model_version):
    model_version = make_model_version(
        metrics={"precision": 0.7, "recall": 0.8, "f1": 0.75, "roc_auc": 0.9}
    )
    resp = client.get(f"/models/{model_version.id}/metrics")
    assert resp.status_code == 200
    assert resp.json()["metrics"]["roc_auc"] == 0.9


def test_get_production_returns_404_when_none_registered(client):
    resp = client.get("/models/production")
    assert resp.status_code == 404


def test_get_production_returns_current_production_model(client, make_model_version):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    resp = client.get("/models/production")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(production.id)


def test_promote_candidate_via_api(client, make_model_version, disable_artifact_check):
    candidate = make_model_version(stage=ModelStage.CANDIDATE)

    resp = client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 200
    assert resp.json()["stage"] == "production"
    assert resp.json()["promoted_at"] is not None

    production_check = client.get("/models/production")
    assert production_check.json()["id"] == str(candidate.id)


def test_promote_via_api_demotes_existing_production_to_previous(
    client, make_model_version, disable_artifact_check
):
    old_production = make_model_version(stage=ModelStage.PRODUCTION)
    candidate = make_model_version(stage=ModelStage.CANDIDATE)

    resp = client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 200

    old_check = client.get(f"/models/{old_production.id}")
    assert old_check.json()["stage"] == "previous"


def test_promote_nonexistent_candidate_returns_404(client):
    resp = client.post(
        "/models/promote", json={"candidate_id": "00000000-0000-0000-0000-000000000000"}
    )
    assert resp.status_code == 404


def test_promote_non_candidate_returns_422(client, make_model_version):
    already_production = make_model_version(stage=ModelStage.PRODUCTION)
    resp = client.post("/models/promote", json={"candidate_id": str(already_production.id)})
    assert resp.status_code == 422


def test_promote_fails_when_artifact_does_not_exist(client, make_model_version):
    """Default policy (MODEL_PROMOTION_REQUIRE_ARTIFACT_CHECK=true): a
    candidate whose mlflow_run_id doesn't correspond to any real MLflow
    run/artifact must be rejected, whether the tracking server is
    unreachable or reachable-but-missing-the-run — both fail safe.
    """
    candidate = make_model_version(
        stage=ModelStage.CANDIDATE, mlflow_run_id="does-not-exist-run-id"
    )
    resp = client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 422
    assert "artifact" in resp.json()["detail"].lower()


def test_promote_missing_metrics_returns_422(client, make_model_version):
    candidate = make_model_version(stage=ModelStage.CANDIDATE, metrics={"precision": 0.5})
    resp = client.post("/models/promote", json={"candidate_id": str(candidate.id)})
    assert resp.status_code == 422
    assert "metric" in resp.json()["detail"].lower()


def test_rollback_returns_422_when_no_previous_model(client, make_model_version):
    make_model_version(stage=ModelStage.PRODUCTION)
    resp = client.post("/models/rollback")
    assert resp.status_code == 422


def test_rollback_restores_previous_production_via_api(client, make_model_version):
    make_model_version(stage=ModelStage.PRODUCTION, version_label="v3")
    previous = make_model_version(stage=ModelStage.PREVIOUS, version_label="v2")

    resp = client.post("/models/rollback")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(previous.id)
    assert resp.json()["stage"] == "production"

    production_check = client.get("/models/production")
    assert production_check.json()["id"] == str(previous.id)
