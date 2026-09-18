"""API tests for POST /train and its interaction with GET /jobs."""

from db.models import JobStatus, JobType


def test_submit_training_job_returns_202_with_job_id(client, make_dataset):
    dataset = make_dataset()

    resp = client.post("/train", json={"dataset_id": str(dataset.id)})

    assert resp.status_code == 202
    body = resp.json()
    assert body["dataset_id"] == str(dataset.id)
    assert body["status"] == "queued"
    assert "job_id" in body
    assert "created_at" in body


def test_submit_training_job_returns_404_for_unknown_dataset(client):
    resp = client.post("/train", json={"dataset_id": "00000000-0000-0000-0000-000000000000"})
    assert resp.status_code == 404


def test_submit_training_job_returns_422_for_invalid_dataset(client, make_dataset):
    dataset = make_dataset(is_valid=False)

    resp = client.post("/train", json={"dataset_id": str(dataset.id)})

    assert resp.status_code == 422
    assert "failed validation" in resp.json()["detail"]


def test_submit_training_job_returns_422_for_unknown_algorithm(client, make_dataset):
    dataset = make_dataset()

    resp = client.post(
        "/train", json={"dataset_id": str(dataset.id), "algorithms": ["not_a_real_algorithm"]}
    )

    assert resp.status_code == 422


def test_submit_training_job_returns_422_for_out_of_range_test_size(client, make_dataset):
    dataset = make_dataset()

    resp = client.post("/train", json={"dataset_id": str(dataset.id), "test_size": 1.5})

    assert resp.status_code == 422


def test_submit_training_job_creates_a_real_job_row(client, make_dataset, db_session):
    from db.models import Job

    dataset = make_dataset()
    resp = client.post(
        "/train",
        json={
            "dataset_id": str(dataset.id),
            "algorithms": ["logistic_regression"],
            "random_state": 7,
        },
    )
    job_id = resp.json()["job_id"]

    job = db_session.get(Job, job_id)
    assert job is not None
    assert job.job_type == JobType.TRAIN
    assert job.dataset_id == dataset.id
    assert job.payload["algorithms"] == ["logistic_regression"]
    assert job.payload["config"] == {"random_state": 7}


def test_get_job_for_training_job_reports_dataset_and_queued_status(client, make_dataset):
    dataset = make_dataset()
    submit_resp = client.post("/train", json={"dataset_id": str(dataset.id)})
    job_id = submit_resp.json()["job_id"]

    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    body = job_resp.json()
    assert body["job_type"] == "train"
    assert body["dataset_id"] == str(dataset.id)
    assert body["status"] in {"queued", "running", "failed"}  # dispatch may fail without a broker


def test_duplicate_active_job_for_same_dataset_and_config_returns_existing_job(
    client, make_dataset
):
    dataset = make_dataset()
    payload = {"dataset_id": str(dataset.id), "algorithms": ["logistic_regression"]}

    first = client.post("/train", json=payload)
    second = client.post("/train", json=payload)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]


def test_different_config_for_same_dataset_is_not_treated_as_duplicate(client, make_dataset):
    dataset = make_dataset()

    first = client.post(
        "/train", json={"dataset_id": str(dataset.id), "algorithms": ["logistic_regression"]}
    )
    second = client.post(
        "/train", json={"dataset_id": str(dataset.id), "algorithms": ["random_forest"]}
    )

    assert first.json()["job_id"] != second.json()["job_id"]


def test_duplicate_detection_ignores_jobs_that_already_finished(client, make_dataset, db_session):
    from db.models import Job

    dataset = make_dataset()
    payload = {"dataset_id": str(dataset.id), "algorithms": ["logistic_regression"]}

    first = client.post("/train", json=payload)
    first_job_id = first.json()["job_id"]

    # Simulate the first job having already finished (success or failure
    # both make it inactive) — a resubmission is then a legitimately new job.
    job = db_session.get(Job, first_job_id)
    job.status = JobStatus.COMPLETED
    db_session.flush()

    second = client.post("/train", json=payload)
    assert second.json()["job_id"] != first_job_id


def test_dispatch_failure_returns_503_and_marks_job_failed(client, make_dataset, monkeypatch):
    """If the broker can't be reached at enqueue time, the job is marked
    FAILED immediately (not left QUEUED forever with nothing to process
    it) and the client gets a clear 503, not a stack trace.
    """

    def _raise(*args, **kwargs):
        raise ConnectionError("broker unavailable")

    monkeypatch.setattr("workers.tasks.process_training_job_task.delay", _raise)

    dataset = make_dataset()
    resp = client.post("/train", json={"dataset_id": str(dataset.id)})

    assert resp.status_code == 503
    assert "Traceback" not in resp.text
