"""Shared pytest fixtures.

Integration tests run against a real local Postgres test database
(predictiq_test) rather than mocks or SQLite — SQLite doesn't support
JSONB/UUID/Enum the same way Postgres does, and a passing test against a
different database engine wouldn't prove the real thing works.

Each test runs inside a transaction that's rolled back afterward, so
tests are isolated without needing to recreate the schema per test.
"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg2://predictiq:predictiq@localhost:5432/predictiq_test"
)
os.environ["ENVIRONMENT"] = "test"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
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
def client(db_session, storage_dir) -> TestClient:
    from app.main import app
    from db.session import get_db

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
