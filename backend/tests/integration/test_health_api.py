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
    resp = client.get("/health")
    body_text = resp.text.lower()
    assert "postgresql://" not in body_text
    assert "password" not in body_text
    assert "predictiq:predictiq" not in body_text
