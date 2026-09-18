"""Integration tests for GET /monitoring/summary, /monitoring/model-usage,
and /monitoring/jobs — all computed from real predictions/jobs/model_versions
rows, never synthesized.
"""

from datetime import UTC, datetime, timedelta

from db.models import Job, JobStatus, JobType, ModelStage


def test_summary_reports_no_production_model_when_none_registered(client):
    resp = client.get("/monitoring/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_production_model"] is None
    assert body["drift_status"] == "no_production_model"
    assert body["recent_prediction_count"] == 0


def test_summary_reports_real_production_model_and_prediction_count(
    client, make_model_version, make_predictions
):
    production = make_model_version(stage=ModelStage.PRODUCTION, version_label="v9")
    make_predictions(production, 7)

    resp = client.get("/monitoring/summary")
    body = resp.json()

    assert body["current_production_model"]["version_label"] == "v9"
    assert body["current_production_model"]["model_version_id"] == str(production.id)
    assert body["recent_prediction_count"] == 7


def test_summary_excludes_predictions_older_than_the_recent_window(
    client, make_model_version, make_predictions
):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    old = datetime.now(UTC) - timedelta(hours=48)
    make_predictions(production, 5, created_at=old)
    make_predictions(production, 3)  # recent

    resp = client.get("/monitoring/summary")
    assert resp.json()["recent_prediction_count"] == 3


def test_summary_includes_active_training_jobs_and_recent_failures(
    client, db_session, make_dataset
):
    dataset = make_dataset()
    running_job = Job(
        job_type=JobType.TRAIN, status=JobStatus.RUNNING, dataset_id=dataset.id, payload={}
    )
    failed_job = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.FAILED,
        dataset_id=dataset.id,
        payload={},
        error_message="deterministic failure for testing",
    )
    db_session.add_all([running_job, failed_job])
    db_session.flush()

    resp = client.get("/monitoring/summary")
    body = resp.json()

    assert body["active_training_jobs"] == 1
    assert len(body["recent_failed_jobs"]) == 1
    assert body["recent_failed_jobs"][0]["error_message"] == "deterministic failure for testing"


def test_summary_includes_model_version_usage_and_job_stats(
    client, make_model_version, make_predictions
):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    make_predictions(production, 4)

    resp = client.get("/monitoring/summary")
    body = resp.json()

    assert "model_version_usage" in body
    assert any(u["model_version_id"] == str(production.id) for u in body["model_version_usage"])
    assert "training_job_stats" in body
    assert body["training_job_stats"]["total_jobs"] >= 0


def test_summary_never_leaks_raw_input_features(client, make_model_version, make_predictions):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    make_predictions(production, 4)

    resp = client.get("/monitoring/summary")
    assert "input_features" not in resp.text


def test_model_usage_reports_prediction_counts_per_version(
    client, make_model_version, make_predictions
):
    v1 = make_model_version(stage=ModelStage.PREVIOUS, version_label="v1")
    v2 = make_model_version(stage=ModelStage.PRODUCTION, version_label="v2")
    make_predictions(v1, 3)
    make_predictions(v2, 7)

    resp = client.get("/monitoring/model-usage")
    assert resp.status_code == 200
    body = resp.json()

    usage_by_label = {u["version_label"]: u for u in body}
    assert usage_by_label["v1"]["prediction_count"] == 3
    assert usage_by_label["v2"]["prediction_count"] == 7
    assert usage_by_label["v1"]["percentage_of_total"] == 30.0
    assert usage_by_label["v2"]["percentage_of_total"] == 70.0


def test_model_usage_shows_stale_model_still_serving_after_promotion(
    client, make_model_version, make_predictions
):
    """The exact scenario section 9 calls out: an old model receiving
    predictions after a newer one was promoted should be visible, with a
    last_prediction_at timestamp that reveals it's still active.
    """
    old_production = make_model_version(stage=ModelStage.PREVIOUS, version_label="v_old")
    new_production = make_model_version(stage=ModelStage.PRODUCTION, version_label="v_new")

    make_predictions(old_production, 2)  # a client hasn't updated yet
    make_predictions(new_production, 10)

    resp = client.get("/monitoring/model-usage")
    body = resp.json()
    stale_entry = next(u for u in body if u["version_label"] == "v_old")

    assert stale_entry["prediction_count"] == 2
    assert stale_entry["last_prediction_at"] is not None


def test_job_stats_reports_real_counts_and_average_duration(client, db_session, make_dataset):
    dataset = make_dataset()
    start = datetime.now(UTC) - timedelta(minutes=5)
    completed_job = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.COMPLETED,
        dataset_id=dataset.id,
        payload={},
        started_at=start,
        finished_at=start + timedelta(seconds=10),
    )
    failed_job = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.FAILED,
        dataset_id=dataset.id,
        payload={},
        error_message="boom",
    )
    db_session.add_all([completed_job, failed_job])
    db_session.flush()

    resp = client.get("/monitoring/jobs")
    body = resp.json()

    assert body["total_jobs"] == 2
    assert body["completed_jobs"] == 1
    assert body["failed_jobs"] == 1
    assert body["average_training_duration_seconds"] == 10.0
    assert len(body["recent_failures"]) == 1
    assert body["recent_failures"][0]["error_message"] == "boom"


def test_job_stats_average_duration_is_none_when_no_completed_jobs(client):
    resp = client.get("/monitoring/jobs")
    body = resp.json()
    assert body["total_jobs"] == 0
    assert body["average_training_duration_seconds"] is None
