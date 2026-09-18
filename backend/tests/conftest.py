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
from db.session import engine


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
        )
        db_session.add(model_version)
        db_session.flush()
        return model_version

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
