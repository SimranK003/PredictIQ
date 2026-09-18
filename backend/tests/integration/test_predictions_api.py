"""API tests for /predict, /predict/batch, /predictions, and /jobs."""

import pytest

from db.models import Job, JobStatus, ModelStage, Prediction
from tests.conftest import FakePipeline, sample_churn_features


def test_predict_returns_prediction_for_valid_input(client, make_model_version, stub_pipeline):
    production = make_model_version(stage=ModelStage.PRODUCTION, version_label="v1")
    stub_pipeline(FakePipeline(positive=True, probability=0.9016))

    resp = client.post("/predict", json=sample_churn_features())

    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "Yes"
    assert body["probability"] == pytest.approx(0.9016)
    assert body["model_version"] == "v1"
    assert body["model_version_id"] == str(production.id)
    assert "prediction_id" in body
    assert "request_id" in body


def test_predict_response_request_id_matches_response_header(
    client, make_model_version, stub_pipeline
):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(positive=False))

    resp = client.post("/predict", json=sample_churn_features())

    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == resp.json()["request_id"]


def test_predict_returns_503_when_no_production_model(client):
    resp = client.post("/predict", json=sample_churn_features())
    assert resp.status_code == 503


def test_predict_returns_422_for_missing_required_field(client, make_model_version, stub_pipeline):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline())

    payload = sample_churn_features()
    del payload["gender"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_returns_422_for_wrong_type(client, make_model_version, stub_pipeline):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline())

    payload = sample_churn_features(tenure="not-a-number")
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_allows_missing_total_charges_for_new_customer(
    client, make_model_version, stub_pipeline
):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(positive=False))

    payload = sample_churn_features(tenure=0)
    del payload["TotalCharges"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_persists_prediction_with_input_features(
    client, make_model_version, stub_pipeline, db_session
):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(positive=True, probability=0.7))

    resp = client.post("/predict", json=sample_churn_features())
    prediction_id = resp.json()["prediction_id"]

    row = db_session.get(Prediction, prediction_id)
    assert row is not None
    assert row.input_features["Contract"] == "Month-to-month"


def test_predict_database_failure_returns_clean_500_without_traceback(
    client, make_model_version, stub_pipeline, monkeypatch
):
    """Starlette's TestClient re-raises unhandled server exceptions by
    default (raise_server_exceptions=True) specifically so tests notice
    real bugs — that's not what we're checking here. We want to verify
    the *response our handler produces*, so this test builds its own
    client (reusing the same app + dependency overrides the `client`
    fixture already configured) with that behavior turned off.
    """
    from starlette.testclient import TestClient

    from app.main import app

    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline())

    def _raise(*args, **kwargs):
        raise RuntimeError("db exploded")

    monkeypatch.setattr("sqlalchemy.orm.Session.commit", _raise)

    lenient_client = TestClient(app, raise_server_exceptions=False)
    resp = lenient_client.post("/predict", json=sample_churn_features())

    assert resp.status_code == 500
    assert "db exploded" not in resp.text
    assert "Traceback" not in resp.text
    assert resp.json()["detail"] == "An internal error occurred."


def test_predict_batch_sync_path_completes_immediately(client, make_model_version, stub_pipeline):
    make_model_version(stage=ModelStage.PRODUCTION, version_label="v2")
    stub_pipeline(FakePipeline(positive=True, probability=0.6))

    records = [sample_churn_features(tenure=i) for i in range(3)]
    resp = client.post("/predict/batch", json={"records": records})

    assert resp.status_code == 202
    body = resp.json()
    assert body["n_records"] == 3
    assert body["model_version"] == "v2"
    assert body["status"] == "completed"


def test_predict_batch_job_retrievable_via_jobs_endpoint(client, make_model_version, stub_pipeline):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(positive=True))

    records = [sample_churn_features() for _ in range(2)]
    batch_resp = client.post("/predict/batch", json={"records": records})
    job_id = batch_resp.json()["job_id"]

    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    body = job_resp.json()
    assert body["status"] == "completed"
    assert body["n_records"] == 2
    assert body["result"]["n_processed"] == 2


def test_predict_batch_exceeding_max_records_returns_400(client, make_model_version, monkeypatch):
    make_model_version(stage=ModelStage.PRODUCTION)
    monkeypatch.setenv("BATCH_PREDICTION_MAX_RECORDS", "2")
    from app.core.config import get_settings

    get_settings.cache_clear()

    records = [sample_churn_features() for _ in range(3)]
    resp = client.post("/predict/batch", json={"records": records})

    get_settings.cache_clear()
    assert resp.status_code == 400


def test_predict_batch_over_sync_threshold_enqueues_async_task(
    client, make_model_version, stub_pipeline, monkeypatch, db_session
):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline())
    monkeypatch.setenv("BATCH_PREDICTION_SYNC_MAX_RECORDS", "1")
    from app.core.config import get_settings

    get_settings.cache_clear()

    enqueued = {}

    def _fake_delay(job_id):
        enqueued["job_id"] = job_id

    monkeypatch.setattr("workers.tasks.process_batch_prediction_task.delay", _fake_delay)

    records = [sample_churn_features() for _ in range(3)]
    resp = client.post("/predict/batch", json={"records": records})

    get_settings.cache_clear()

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert enqueued["job_id"] == body["job_id"]

    job = db_session.get(Job, body["job_id"])
    assert job.status == JobStatus.QUEUED


def test_get_job_returns_404_for_unknown_job(client):
    resp = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_list_predictions_pagination(client, make_model_version, stub_pipeline):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline())
    for i in range(5):
        client.post("/predict", json=sample_churn_features(tenure=i))

    page1 = client.get("/predictions", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/predictions", params={"limit": 2, "offset": 2}).json()

    assert page1["total"] == 5
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 2
    assert {p["id"] for p in page1["items"]}.isdisjoint({p["id"] for p in page2["items"]})


def test_list_predictions_filter_by_model_version(client, make_model_version, stub_pipeline):
    v1 = make_model_version(stage=ModelStage.PRODUCTION, version_label="v1")
    stub_pipeline(FakePipeline())
    client.post("/predict", json=sample_churn_features())

    resp = client.get("/predictions", params={"model_version_id": str(v1.id)})
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["model_version_id"] == str(v1.id)


def test_get_prediction_by_id_returns_detail_with_model_metadata(
    client, make_model_version, stub_pipeline
):
    make_model_version(stage=ModelStage.PRODUCTION, version_label="v3", algorithm="random_forest")
    stub_pipeline(FakePipeline(positive=True, probability=0.55))

    predict_resp = client.post("/predict", json=sample_churn_features())
    prediction_id = predict_resp.json()["prediction_id"]

    resp = client.get(f"/predictions/{prediction_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version_label"] == "v3"
    assert body["algorithm"] == "random_forest"
    assert body["input_features"]["gender"] == "Female"


def test_get_prediction_by_id_returns_404_when_missing(client):
    resp = client.get("/predictions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_metrics_endpoint_reflects_real_predictions(client, make_model_version, stub_pipeline):
    make_model_version(stage=ModelStage.PRODUCTION, version_label="v-metrics")
    stub_pipeline(FakePipeline(positive=True))

    client.post("/predict", json=sample_churn_features())
    client.post("/predict", json=sample_churn_features())

    metrics_text = client.get("/metrics").text
    expected_series = (
        'predictiq_predictions_total{algorithm="logistic_regression",model_version="v-metrics"}'
    )
    assert expected_series in metrics_text
    assert "predictiq_http_requests_total" in metrics_text
    assert "predictiq_http_request_latency_seconds" in metrics_text


# Drift-specific tests moved to tests/integration/test_monitoring_drift_api.py
# (Phase 6 — real statistical drift detection superseding the Phase 4
# descriptive-stats-only foundation).
