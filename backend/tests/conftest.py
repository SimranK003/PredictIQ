"""Shared pytest fixtures.

Integration tests run against a real local Postgres test database
(predictiq_test) rather than mocks or SQLite — SQLite doesn't support
JSONB/UUID/Enum the same way Postgres does, and a passing test against a
different database engine wouldn't prove the real thing works.

Each test runs inside a transaction that's rolled back afterward, so
tests are isolated without needing to recreate the schema per test.
"""

import hashlib
import os
import uuid

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg2://predictiq:predictiq@localhost:5432/predictiq_test"
)
os.environ["ENVIRONMENT"] = "test"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from db.models import Dataset, ModelStage, ModelVersionRecord
from db.session import SessionLocal, engine


@pytest.fixture(scope="session", autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def db_session() -> Session:
    connection = engine.connect()
    transaction = connection.begin()
    TestSessionLocal = sessionmaker(bind=connection, autoflush=False, autocommit=False, future=True)
    session = TestSessionLocal()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def storage_dir(tmp_path, monkeypatch):
    path = tmp_path / "dataset_storage"
    monkeypatch.setenv("DATASET_STORAGE_DIR", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture()
def make_dataset(db_session):
    """Factory for a valid (or deliberately invalid) Dataset row, for
    tests that need one to satisfy model_versions.dataset_id.
    """

    def _make(*, is_valid: bool = True, **overrides) -> Dataset:
        dataset = Dataset(
            filename=overrides.get("filename", "test_dataset.csv"),
            schema_name=overrides.get("schema_name", "telco_customer_churn"),
            storage_path=overrides.get("storage_path", "/tmp/test_dataset.csv"),
            content_hash=overrides.get(
                "content_hash", hashlib.sha256(uuid.uuid4().bytes).hexdigest()
            ),
            n_rows=overrides.get("n_rows", 100),
            n_columns=overrides.get("n_columns", 21),
            is_valid=is_valid,
            quality_report=overrides.get("quality_report", {"issues": [], "is_valid": is_valid}),
        )
        db_session.add(dataset)
        db_session.flush()
        return dataset

    return _make


@pytest.fixture()
def mlflow_tmp_tracking_env(tmp_path, monkeypatch):
    """Points MLFLOW_TRACKING_URI at a local file store for the duration
    of a test, via the same env-var + settings-cache-clear mechanism
    configure_mlflow() itself reads — so code under test that calls
    configure_mlflow() (e.g. train_and_register) picks this up instead
    of needing a live tracking server.
    """
    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    get_settings.cache_clear()
    yield tracking_uri
    get_settings.cache_clear()


@pytest.fixture()
def make_real_dataset_file(tmp_path, make_dataset):
    """Writes a real small CSV to disk and creates a matching Dataset
    row with the file's actual sha256 — for tests that exercise real
    training (execute_training_job re-reads and re-hashes the file, so
    a fixture that fabricates a hash without a real file won't do).
    """
    from tests.integration.test_training_pipeline import _synthetic_churn_dataframe

    def _make(*, n_rows: int = 300, seed: int = 5, **overrides) -> Dataset:
        df = _synthetic_churn_dataframe(n=n_rows, seed=seed)
        csv_path = tmp_path / f"dataset_{uuid.uuid4().hex}.csv"
        df.to_csv(csv_path, index=False)
        content_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()

        return make_dataset(
            storage_path=str(csv_path),
            content_hash=content_hash,
            n_rows=len(df),
            n_columns=len(df.columns),
            is_valid=overrides.pop("is_valid", True),
            **overrides,
        )

    return _make


@pytest.fixture()
def make_model_version(db_session, make_dataset):
    """Factory for a ModelVersionRecord, defaulting to a well-formed
    candidate with all metrics the default promotion policy requires.
    """

    def _make(
        *, stage: ModelStage = ModelStage.CANDIDATE, dataset: Dataset | None = None, **overrides
    ):
        ds = dataset or make_dataset()
        model_version = ModelVersionRecord(
            version_label=overrides.get("version_label", f"v{uuid.uuid4().hex[:6]}"),
            algorithm=overrides.get("algorithm", "logistic_regression"),
            mlflow_run_id=overrides.get("mlflow_run_id", uuid.uuid4().hex),
            mlflow_experiment_id=overrides.get("mlflow_experiment_id", "1"),
            artifact_uri=overrides.get("artifact_uri", "runs:/fake-run-id/model"),
            git_commit=overrides.get("git_commit", "abc123def456"),
            stage=stage,
            metrics=overrides.get(
                "metrics", {"precision": 0.55, "recall": 0.70, "f1": 0.61, "roc_auc": 0.84}
            ),
            params=overrides.get("params", {"C": 1.0}),
            dataset_id=ds.id,
            training_job_id=overrides.get("training_job_id"),
        )
        db_session.add(model_version)
        db_session.flush()
        return model_version

    return _make


class FakePipeline:
    """Stand-in for a fitted sklearn Pipeline in tests that exercise our
    own orchestration (persistence, error handling, metrics) rather than
    sklearn/MLflow itself, which is already covered in Phase 2/3 tests
    against real artifacts.
    """

    def __init__(
        self,
        *,
        positive: bool = True,
        probability: float | None = None,
        error: Exception | None = None,
    ):
        self.positive = positive
        self.probability = probability if probability is not None else (0.9 if positive else 0.1)
        self.error = error

    def predict(self, df):
        import numpy as np

        if self.error:
            raise self.error
        return np.array([1 if self.positive else 0] * len(df))

    def predict_proba(self, df):
        import numpy as np

        if self.error:
            raise self.error
        return np.array([[1 - self.probability, self.probability]] * len(df))


@pytest.fixture()
def stub_pipeline(monkeypatch):
    """Patch mlflow.sklearn.load_model (used by both the production
    cache and the pinned-batch loader) to return a FakePipeline, and
    reset the module-level production cache before/after so tests don't
    leak a loaded model into each other.
    """
    from app.services import model_loader

    model_loader.invalidate_production_model_cache()

    def _stub(pipeline: FakePipeline):
        monkeypatch.setattr(model_loader.mlflow.sklearn, "load_model", lambda uri: pipeline)
        return pipeline

    yield _stub
    model_loader.invalidate_production_model_cache()


def sample_churn_features(**overrides) -> dict:
    """A valid ChurnFeaturesIn-shaped dict, for tests that need a
    realistic prediction request body without hand-rolling all 19 fields
    every time.
    """
    base = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def real_committing_session():
    """A session on its own real connection whose commits are genuinely
    durable — needed for tests exercising code that itself calls
    db.rollback() (execute_training_job's and execute_batch_prediction's
    failure paths). Inside the shared db_session fixture, an internal
    rollback would undo the test's own setup too, since nothing there is
    ever really committed to Postgres until the test ends (see
    db_session's docstring) — this fixture avoids that by being real.
    No auto-cleanup: tests using it call cleanup_training_artifacts (or
    equivalent) explicitly, since what needs deleting varies per test.
    """
    session = SessionLocal()
    yield session
    session.close()


def cleanup_training_artifacts(session, *, job_id=None, dataset_id=None) -> None:
    """Deletes a real training job's rows in FK-safe order. Shared by
    tests that create real Dataset/Job/ModelVersionRecord rows via
    real_committing_session and must clean up after themselves.
    """
    from db.models import Job, ModelLifecycleEvent, ModelVersionRecord, Prediction

    if job_id is not None:
        candidate_ids = [
            row.id
            for row in session.query(ModelVersionRecord.id)
            .filter(ModelVersionRecord.training_job_id == job_id)
            .all()
        ]
        if candidate_ids:
            session.query(ModelLifecycleEvent).filter(
                ModelLifecycleEvent.model_version_id.in_(candidate_ids)
            ).delete(synchronize_session=False)
            session.query(Prediction).filter(
                Prediction.model_version_id.in_(candidate_ids)
            ).delete(synchronize_session=False)
            session.query(ModelVersionRecord).filter(
                ModelVersionRecord.id.in_(candidate_ids)
            ).delete(synchronize_session=False)
        session.query(Job).filter(Job.id == job_id).delete(synchronize_session=False)
    if dataset_id is not None:
        session.query(Dataset).filter(Dataset.id == dataset_id).delete(synchronize_session=False)
    session.commit()


@pytest.fixture()
def make_training_job(db_session):
    """Factory for a queued TRAIN Job row, defaulting to a config/
    algorithm selection that trains fast (small forests, no XGBoost —
    XGBoost is the slowest of the three on a small synthetic dataset and
    most job-lifecycle tests don't need all three algorithms to prove
    their point).
    """
    from db.models import Job, JobStatus, JobType

    def _make(*, dataset: Dataset, config: dict | None = None, algorithms=None, **overrides):
        job = Job(
            job_type=JobType.TRAIN,
            status=overrides.get("status", JobStatus.QUEUED),
            dataset_id=dataset.id,
            payload={
                "config": config if config is not None else {"random_forest_n_estimators": 10},
                "algorithms": algorithms if algorithms is not None else ["logistic_regression"],
            },
        )
        db_session.add(job)
        db_session.flush()
        return job

    return _make


@pytest.fixture()
def make_predictions(db_session):
    """Bulk-creates real Prediction rows tied to a model version, for
    drift/monitoring tests that need many observations. `feature_fn(i)`
    generates the input_features dict for row i; `created_at` can
    backdate all rows for time-window tests (server_default only applies
    when the column is omitted, so an explicit value here is honored).
    """
    from db.models import Prediction

    def _make(model_version, n: int, *, feature_fn=None, created_at=None, prediction="No"):
        feature_fn = feature_fn or (lambda i: sample_churn_features(tenure=i % 72))
        rows = []
        for i in range(n):
            pred = Prediction(
                request_id=uuid.uuid4(),
                model_version_id=model_version.id,
                input_features=feature_fn(i),
                prediction=prediction,
                probability=0.3,
                latency_ms=5.0,
            )
            if created_at is not None:
                pred.created_at = created_at
            db_session.add(pred)
            rows.append(pred)
        db_session.flush()
        return rows

    return _make


@pytest.fixture()
def client(db_session, storage_dir) -> TestClient:
    from app.main import app
    from db.session import get_db

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
