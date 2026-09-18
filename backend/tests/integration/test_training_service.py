"""Tests for app/services/training.py — the async training job lifecycle.

Uses REAL training (small synthetic dataset, real sklearn/xgboost
fitting, real file-store MLflow) rather than stubbing the ML pipeline,
because these tests specifically verify lineage (dataset -> job -> MLflow
run -> model version) — mocking any link in that chain would prove the
wrong thing. Real Postgres throughout; no mocked DB behavior.

Tests whose code path hits execute_training_job's internal db.rollback()
(the deterministic-failure branch) use real_committing_session instead
of the shared db_session fixture — see conftest.py's docstring on why a
rollback inside the shared fixture's single transaction would undo the
test's own setup, not just the failed work.
"""

import hashlib
import os
import uuid

import pytest

from app.services.training import TrainingExecutionError, execute_training_job
from db.models import Dataset, Job, JobStatus, JobType, ModelVersionRecord
from tests.conftest import cleanup_training_artifacts


def test_execute_training_job_completes_and_registers_candidates(
    db_session, make_real_dataset_file, make_training_job, mlflow_tmp_tracking_env
):
    dataset = make_real_dataset_file()
    job = make_training_job(dataset=dataset, algorithms=["logistic_regression", "random_forest"])

    execute_training_job(db_session, job.id)
    db_session.refresh(job)

    assert job.status == JobStatus.COMPLETED
    assert job.started_at is not None
    assert job.finished_at is not None
    assert set(job.result["algorithms_trained"]) == {"logistic_regression", "random_forest"}
    assert job.result["best_algorithm"] in {"logistic_regression", "random_forest"}
    assert len(job.result["candidate_model_version_ids"]) == 2


def test_completed_training_job_produces_real_candidates_linked_to_job(
    db_session, make_real_dataset_file, make_training_job, mlflow_tmp_tracking_env
):
    """Full lineage check: Model Version -> Training Job -> Dataset."""
    dataset = make_real_dataset_file()
    job = make_training_job(dataset=dataset)

    execute_training_job(db_session, job.id)

    candidates = (
        db_session.query(ModelVersionRecord)
        .filter(ModelVersionRecord.training_job_id == job.id)
        .all()
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.dataset_id == dataset.id
    assert candidate.training_job_id == job.id
    assert candidate.mlflow_run_id  # a real run id was assigned
    assert candidate.git_commit  # populated, even if "unknown" outside a git repo
    assert candidate.metrics["roc_auc"] is not None
    assert "test_size" in candidate.params
    assert "preprocessing" in candidate.params


def test_verify_dataset_for_training_raises_for_nonexistent_dataset(db_session):
    from app.services.training import _verify_dataset_for_training

    fake_dataset_id = uuid.uuid4()

    with pytest.raises(TrainingExecutionError, match="not found"):
        _verify_dataset_for_training(db_session, fake_dataset_id)


def test_execute_training_job_propagates_transient_exceptions_without_marking_failed(
    db_session, make_real_dataset_file, make_training_job, mlflow_tmp_tracking_env, monkeypatch
):
    """Transient infra errors must NOT be swallowed into a FAILED status
    — they propagate so the Celery task wrapper can retry (see
    workers/tasks.py); the job is left at RUNNING for the retry to pick
    back up. No internal db.rollback() happens on this path, so the
    shared db_session fixture is fine here.
    """
    import requests

    dataset = make_real_dataset_file()
    job = make_training_job(dataset=dataset)

    def _raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("mlflow unreachable")

    monkeypatch.setattr("app.services.training.train_and_register", _raise_connection_error)

    with pytest.raises(requests.exceptions.ConnectionError):
        execute_training_job(db_session, job.id)

    db_session.refresh(job)
    assert job.status == JobStatus.RUNNING
    assert job.error_message is None


# --- Failure-path tests: real committing session (see module docstring) ---


def _real_dataset(session, tmp_path, *, is_valid: bool = True, n_rows: int = 300) -> Dataset:
    from tests.integration.test_training_pipeline import _synthetic_churn_dataframe

    df = _synthetic_churn_dataframe(n=n_rows, seed=5)
    csv_path = tmp_path / f"dataset_{uuid.uuid4().hex}.csv"
    df.to_csv(csv_path, index=False)
    content_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    dataset = Dataset(
        filename="real_training_service_test.csv",
        schema_name="telco_customer_churn",
        storage_path=str(csv_path),
        content_hash=content_hash,
        n_rows=len(df),
        n_columns=len(df.columns),
        is_valid=is_valid,
        quality_report={"issues": []},
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


def _real_training_job(session, dataset: Dataset, *, algorithms=None, config=None) -> Job:
    job = Job(
        job_type=JobType.TRAIN,
        status=JobStatus.QUEUED,
        dataset_id=dataset.id,
        payload={
            "config": config or {"random_forest_n_estimators": 10},
            "algorithms": algorithms or ["logistic_regression"],
        },
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def test_execute_training_job_marks_failed_for_invalid_dataset(
    real_committing_session, tmp_path
):
    session = real_committing_session
    dataset = _real_dataset(session, tmp_path, is_valid=False)
    job = _real_training_job(session, dataset)

    try:
        execute_training_job(session, job.id)
        session.refresh(job)

        assert job.status == JobStatus.FAILED
        assert "failed validation" in job.error_message
        assert (
            session.query(ModelVersionRecord)
            .filter(ModelVersionRecord.training_job_id == job.id)
            .count()
            == 0
        )
    finally:
        cleanup_training_artifacts(session, job_id=job.id, dataset_id=dataset.id)


def test_execute_training_job_marks_failed_for_content_hash_mismatch(
    real_committing_session, tmp_path
):
    session = real_committing_session
    dataset = _real_dataset(session, tmp_path)
    dataset.content_hash = "0" * 64
    session.commit()
    job = _real_training_job(session, dataset)

    try:
        execute_training_job(session, job.id)
        session.refresh(job)

        assert job.status == JobStatus.FAILED
        assert "hash mismatch" in job.error_message
    finally:
        cleanup_training_artifacts(session, job_id=job.id, dataset_id=dataset.id)


def test_execute_training_job_marks_failed_for_missing_file_on_disk(
    real_committing_session, tmp_path
):
    session = real_committing_session
    dataset = _real_dataset(session, tmp_path)
    os.remove(dataset.storage_path)
    job = _real_training_job(session, dataset)

    try:
        execute_training_job(session, job.id)
        session.refresh(job)

        assert job.status == JobStatus.FAILED
        assert "missing on disk" in job.error_message
    finally:
        cleanup_training_artifacts(session, job_id=job.id, dataset_id=dataset.id)


def test_execute_training_job_marks_failed_for_deterministic_training_error(
    real_committing_session, tmp_path, mlflow_tmp_tracking_env
):
    """An unknown algorithm name is a deterministic config error — must
    fail immediately, not be retried.
    """
    session = real_committing_session
    dataset = _real_dataset(session, tmp_path)
    job = _real_training_job(session, dataset, algorithms=["not_a_real_algorithm"])

    try:
        execute_training_job(session, job.id)
        session.refresh(job)

        assert job.status == JobStatus.FAILED
        assert "Unknown algorithm" in job.error_message
    finally:
        cleanup_training_artifacts(session, job_id=job.id, dataset_id=dataset.id)


def test_partial_candidates_recorded_when_training_fails_after_some_succeed(
    real_committing_session, tmp_path, mlflow_tmp_tracking_env, monkeypatch
):
    """If train_and_register registers some candidates and then raises
    (e.g. algorithm #2 of N fails after #1 succeeded), the job is FAILED
    but the already-real, already-logged-to-MLflow candidate from #1 is
    not deleted — and the job's result records which ones survived.
    """
    session = real_committing_session
    dataset = _real_dataset(session, tmp_path)
    job = _real_training_job(
        session, dataset, algorithms=["logistic_regression", "random_forest"]
    )

    from app.services.registry import register_candidate as real_register_candidate

    call_count = {"n": 0}

    def _register_then_fail(db, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_register_candidate(db, **kwargs)
        raise RuntimeError("simulated failure registering the second candidate")

    monkeypatch.setattr("ml.training.train.register_candidate", _register_then_fail)

    try:
        execute_training_job(session, job.id)
        session.refresh(job)

        assert job.status == JobStatus.FAILED
        survivors = (
            session.query(ModelVersionRecord)
            .filter(ModelVersionRecord.training_job_id == job.id)
            .all()
        )
        assert len(survivors) == 1
        assert job.result["partial_candidate_model_version_ids"] == [str(survivors[0].id)]
    finally:
        cleanup_training_artifacts(session, job_id=job.id, dataset_id=dataset.id)
