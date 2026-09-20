"""Tests for GET /jobs (list) — added in Phase 7 to support the /jobs
dashboard page, mirroring the existing /datasets and /models list+detail
pattern. GET /jobs/{id} itself is already covered by earlier phases'
tests; these focus on the new list/filter/pagination behavior.
"""

from datetime import UTC, datetime, timedelta

from db.models import Job, JobStatus, JobType


def test_list_jobs_empty(client):
    resp = client.get("/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_list_jobs_returns_real_rows_newest_first(client, db_session, make_dataset):
    # Postgres's now() is transaction-start time, so two rows flushed in
    # the same test transaction would otherwise tie on created_at —
    # backdating "older" makes the ordering assertion meaningful rather
    # than accidentally passing/failing on insertion order.
    dataset = make_dataset()
    older = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.COMPLETED,
        dataset_id=dataset.id,
        payload={},
        created_at=datetime.now(UTC) - timedelta(hours=1),
    )
    db_session.add(older)
    db_session.flush()
    newer = Job(job_type=JobType.TRAIN, status=JobStatus.QUEUED, dataset_id=dataset.id, payload={})
    db_session.add(newer)
    db_session.flush()

    resp = client.get("/jobs")
    body = resp.json()

    assert body["total"] == 2
    assert body["items"][0]["id"] == str(newer.id)
    assert body["items"][1]["id"] == str(older.id)


def test_list_jobs_filters_by_job_type(client, db_session, make_dataset):
    dataset = make_dataset()
    train_job = Job(
        job_type=JobType.TRAIN, status=JobStatus.COMPLETED, dataset_id=dataset.id, payload={}
    )
    batch_job = Job(
        job_type=JobType.BATCH_PREDICT, status=JobStatus.COMPLETED, payload={"records": []}
    )
    db_session.add_all([train_job, batch_job])
    db_session.flush()

    resp = client.get("/jobs", params={"job_type": "train"})
    body = resp.json()

    assert body["total"] == 1
    assert body["items"][0]["id"] == str(train_job.id)


def test_list_jobs_filters_by_status(client, db_session, make_dataset):
    dataset = make_dataset()
    queued = Job(job_type=JobType.TRAIN, status=JobStatus.QUEUED, dataset_id=dataset.id, payload={})
    failed = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.FAILED,
        dataset_id=dataset.id,
        payload={},
        error_message="boom",
    )
    db_session.add_all([queued, failed])
    db_session.flush()

    resp = client.get("/jobs", params={"status": "failed"})
    body = resp.json()

    assert body["total"] == 1
    assert body["items"][0]["id"] == str(failed.id)
    assert body["items"][0]["error_message"] == "boom"


def test_list_jobs_pagination(client, db_session, make_dataset):
    dataset = make_dataset()
    for _ in range(5):
        db_session.add(
            Job(
                job_type=JobType.TRAIN,
                status=JobStatus.COMPLETED,
                dataset_id=dataset.id,
                payload={},
            )
        )
    db_session.flush()

    page1 = client.get("/jobs", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/jobs", params={"limit": 2, "offset": 2}).json()

    assert page1["total"] == 5
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 2
    ids_1 = {j["id"] for j in page1["items"]}
    ids_2 = {j["id"] for j in page2["items"]}
    assert ids_1.isdisjoint(ids_2)


def test_list_jobs_reports_model_version_ids_for_training_jobs(
    client, db_session, make_dataset, make_model_version
):
    dataset = make_dataset()
    job = Job(job_type=JobType.TRAIN, status=JobStatus.COMPLETED, dataset_id=dataset.id, payload={})
    db_session.add(job)
    db_session.flush()
    candidate = make_model_version(dataset=dataset, training_job_id=job.id)

    resp = client.get("/jobs")
    body = resp.json()

    assert body["items"][0]["model_version_ids"] == [str(candidate.id)]
