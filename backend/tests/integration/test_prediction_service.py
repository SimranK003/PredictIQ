"""Tests for app/services/prediction.py against real Postgres.

The ML pipeline itself is stubbed (FakePipeline) — we're proving our own
orchestration (persistence, exact model-version attribution, error
handling, metrics), not sklearn/MLflow correctness, which Phase 2/3
already cover against real artifacts.
"""

import uuid

import pytest

from app.core.exceptions import NotFoundError
from app.services.prediction import (
    PredictionExecutionError,
    execute_batch_prediction,
    run_single_prediction,
)
from db.models import Dataset, Job, JobStatus, JobType, ModelStage, ModelVersionRecord, Prediction
from db.session import SessionLocal
from tests.conftest import FakePipeline, sample_churn_features


def test_run_single_prediction_persists_and_returns_expected_result(
    db_session, make_model_version, stub_pipeline
):
    production = make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(positive=True, probability=0.9016))
    request_id = uuid.uuid4()

    result = run_single_prediction(db_session, sample_churn_features(), request_id=request_id)

    assert result.prediction_row.prediction == "Yes"
    assert result.prediction_row.probability == pytest.approx(0.9016)
    assert result.prediction_row.model_version_id == production.id
    assert result.prediction_row.request_id == request_id

    persisted = db_session.get(Prediction, result.prediction_row.id)
    assert persisted is not None
    assert persisted.input_features["tenure"] == 2


def test_prediction_records_exact_model_version_across_a_promotion(
    db_session, make_model_version, stub_pipeline
):
    """Prediction A -> model v1, Prediction B -> model v2 — both remain
    historically attributable to the exact version that produced them,
    even after v1 is no longer production.
    """
    v1 = make_model_version(stage=ModelStage.PRODUCTION, version_label="v1")
    stub_pipeline(FakePipeline(positive=False, probability=0.2))
    result_a = run_single_prediction(db_session, sample_churn_features(), request_id=uuid.uuid4())

    # Simulate promotion: v1 -> previous, v2 -> production.
    v1.stage = ModelStage.PREVIOUS
    db_session.flush()
    v2 = make_model_version(stage=ModelStage.PRODUCTION, version_label="v2")
    from app.services import model_loader

    model_loader.invalidate_production_model_cache()
    stub_pipeline(FakePipeline(positive=True, probability=0.8))
    result_b = run_single_prediction(db_session, sample_churn_features(), request_id=uuid.uuid4())

    assert result_a.prediction_row.model_version_id == v1.id
    assert result_b.prediction_row.model_version_id == v2.id
    assert result_a.prediction_row.model_version_id != result_b.prediction_row.model_version_id

    # v1's prediction stays attributed to v1 even though v1 is no longer production.
    reloaded_a = db_session.get(Prediction, result_a.prediction_row.id)
    assert reloaded_a.model_version_id == v1.id


def test_run_single_prediction_raises_not_found_when_no_production_model(db_session):
    with pytest.raises(NotFoundError):
        run_single_prediction(db_session, sample_churn_features(), request_id=uuid.uuid4())


def test_run_single_prediction_does_not_persist_when_pipeline_raises(
    db_session, make_model_version, stub_pipeline
):
    make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(error=ValueError("boom")))

    with pytest.raises(PredictionExecutionError):
        run_single_prediction(db_session, sample_churn_features(), request_id=uuid.uuid4())

    assert db_session.query(Prediction).count() == 0


def test_execute_batch_prediction_processes_all_records_and_completes(
    db_session, make_model_version, stub_pipeline
):
    model_version = make_model_version(stage=ModelStage.PRODUCTION)
    stub_pipeline(FakePipeline(positive=True, probability=0.75))

    records = [sample_churn_features(tenure=i) for i in range(5)]
    job = Job(
        job_type=JobType.BATCH_PREDICT,
        status=JobStatus.QUEUED,
        payload={"records": records, "model_version_id": str(model_version.id)},
    )
    db_session.add(job)
    db_session.flush()

    execute_batch_prediction(db_session, job.id)
    db_session.refresh(job)

    assert job.status == JobStatus.COMPLETED
    assert job.result["n_processed"] == 5
    assert len(job.result["prediction_ids"]) == 5
    assert job.finished_at is not None

    predictions = db_session.query(Prediction).filter(Prediction.request_id == job.id).all()
    assert len(predictions) == 5
    assert all(p.model_version_id == model_version.id for p in predictions)
    assert all(p.prediction == "Yes" for p in predictions)


def test_execute_batch_prediction_marks_job_failed_on_pipeline_error(stub_pipeline):
    """Uses a real, genuinely-committing session rather than the shared
    rollback-per-test fixture: execute_batch_prediction's exception path
    calls db.rollback(), which — inside the shared fixture's single
    savepoint-less transaction — would undo the job's own creation too
    (nothing in that fixture is ever really committed to Postgres until
    the test ends). In real usage the job row is committed by the router
    before this function is ever called, on its own transaction, so a
    later rollback here only undoes this function's own partial work.
    This test mirrors that real sequencing, then cleans up explicitly.
    """
    stub_pipeline(FakePipeline(error=RuntimeError("pipeline exploded")))

    session = SessionLocal()
    dataset = Dataset(
        filename="batch_failure_test.csv",
        schema_name="telco_customer_churn",
        storage_path="/tmp/batch_failure_test.csv",
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        n_rows=10,
        n_columns=21,
        is_valid=True,
        quality_report={"issues": []},
    )
    session.add(dataset)
    session.flush()

    model_version = ModelVersionRecord(
        version_label="batch-failure-test-v1",
        algorithm="logistic_regression",
        mlflow_run_id=f"batch-failure-{uuid.uuid4().hex}",
        mlflow_experiment_id="1",
        artifact_uri="runs:/fake/model",
        git_commit="deadbeef",
        stage=ModelStage.PRODUCTION,
        metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5},
        params={},
        dataset_id=dataset.id,
    )
    session.add(model_version)
    session.commit()

    job = Job(
        job_type=JobType.BATCH_PREDICT,
        status=JobStatus.QUEUED,
        payload={"records": [sample_churn_features()], "model_version_id": str(model_version.id)},
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    try:
        execute_batch_prediction(session, job.id)
        session.refresh(job)

        assert job.status == JobStatus.FAILED
        assert "pipeline exploded" in job.error_message
        assert session.query(Prediction).filter(Prediction.request_id == job.id).count() == 0
    finally:
        session.query(Prediction).filter(Prediction.request_id == job.id).delete()
        session.query(Job).filter(Job.id == job.id).delete()
        session.query(ModelVersionRecord).filter(ModelVersionRecord.id == model_version.id).delete()
        session.query(Dataset).filter(Dataset.id == dataset.id).delete()
        session.commit()
        session.close()
