"""Health endpoint tests.

No secrets/connection strings should ever appear in the response — only
ok/unavailable per dependency, plus enough model metadata to be useful.
"""


def test_health_is_degraded_when_no_production_model_registered(client):
    """A fresh install with no promoted model yet: API and DB are fine,
    but the service can't actually serve predictions — "degraded," not
    "ok," and still 200 since most of the API works.
    """
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database"] == "ok"
    assert body["production_model"]["available"] is False


def test_health_is_ok_when_production_model_registered(client, make_model_version):
    from db.models import ModelStage

    make_model_version(stage=ModelStage.PRODUCTION, version_label="v7", algorithm="xgboost")

    resp = client.get("/health")
    body = resp.json()
    assert body["production_model"]["available"] is True
    assert body["production_model"]["version"] == "v7"
    assert body["production_model"]["algorithm"] == "xgboost"
    # status may still be "degraded" if redis is unreachable in this env;
    # what matters here is that the model fields are populated correctly.


def test_health_never_leaks_connection_details(client):
    from app.core.config import get_settings

    resp = client.get("/health")
    body_text = resp.text.lower()
    assert "postgresql://" not in body_text
    assert "password" not in body_text

    # Check against the *actual* configured credential rather than a
    # hardcoded string, so this keeps testing something real even if
    # the local test database's credential ever changes (see
    # tests/conftest.py) instead of silently going stale.
    db_url = get_settings().database_url
    if "://" in db_url and "@" in db_url:
        credentials_part = db_url.split("://", 1)[1].split("@", 1)[0]
        assert credentials_part.lower() not in body_text
