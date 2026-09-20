"""app/services/dataset_storage.py: local-backend behavior is exercised
directly (no mocking needed — it's just Path.write_bytes/read_bytes,
already exhausted by the existing dataset ingestion/training tests via
the real filesystem). This file's job is the S3 backend, which has no
real cloud credentials available in this environment — see README's
Production deployment section on what could and couldn't be verified —
so it's tested against a small in-memory fake S3 client rather than
real AWS/R2. The fake gets the request shape (Bucket/Key/Body) right,
not real S3 semantics (auth, multipart, eventual consistency, etc.).
"""

import uuid

import pytest

from app.core.config import get_settings
from app.services import dataset_storage


class FakeS3Client:
    """In-memory stand-in for boto3's S3 client — enough surface for
    put_object/get_object/head_object as this module calls them.
    """

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"no such object: s3://{Bucket}/{Key}")
        return {"Body": _FakeBody(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"no such object: s3://{Bucket}/{Key}")
        return {}


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


@pytest.fixture()
def s3_backend(monkeypatch):
    """Switches DATASET_STORAGE_BACKEND to s3 with a fake bucket/client
    for the duration of a test.
    """
    monkeypatch.setenv("DATASET_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("DATASET_STORAGE_S3_BUCKET", "predictiq-test-bucket")
    monkeypatch.setenv("DATASET_STORAGE_S3_PREFIX", "datasets")
    get_settings.cache_clear()

    fake_client = FakeS3Client()
    monkeypatch.setattr(dataset_storage, "_s3_client", lambda: fake_client)

    yield fake_client
    get_settings.cache_clear()


def test_local_backend_is_the_default(monkeypatch):
    monkeypatch.delenv("DATASET_STORAGE_BACKEND", raising=False)
    get_settings.cache_clear()
    assert get_settings().dataset_storage_backend == "local"
    get_settings.cache_clear()


def test_save_dataset_bytes_local_writes_to_disk_and_returns_a_local_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("DATASET_STORAGE_BACKEND", raising=False)
    get_settings.cache_clear()

    dataset_id = uuid.uuid4()
    storage_path = dataset_storage.save_dataset_bytes(dataset_id, b"a,b\n1,2\n")

    assert not storage_path.startswith("s3://")
    assert dataset_storage.dataset_file_exists(storage_path)
    assert dataset_storage.read_dataset_bytes(storage_path) == b"a,b\n1,2\n"
    get_settings.cache_clear()


def test_save_dataset_bytes_s3_uploads_and_returns_an_s3_uri(s3_backend):
    dataset_id = uuid.uuid4()
    storage_path = dataset_storage.save_dataset_bytes(dataset_id, b"a,b\n1,2\n")

    assert storage_path == f"s3://predictiq-test-bucket/datasets/{dataset_id}.csv"
    stored = s3_backend.objects[("predictiq-test-bucket", f"datasets/{dataset_id}.csv")]
    assert stored == b"a,b\n1,2\n"


def test_dataset_file_exists_s3_true_for_uploaded_object(s3_backend):
    dataset_id = uuid.uuid4()
    storage_path = dataset_storage.save_dataset_bytes(dataset_id, b"x")
    assert dataset_storage.dataset_file_exists(storage_path) is True


def test_dataset_file_exists_s3_false_for_missing_object(s3_backend):
    missing_uri = "s3://predictiq-test-bucket/datasets/missing.csv"
    assert dataset_storage.dataset_file_exists(missing_uri) is False


def test_read_dataset_bytes_s3_round_trips(s3_backend):
    dataset_id = uuid.uuid4()
    storage_path = dataset_storage.save_dataset_bytes(dataset_id, b"col1,col2\n1,2\n")
    assert dataset_storage.read_dataset_bytes(storage_path) == b"col1,col2\n1,2\n"


def test_read_dataset_dataframe_s3_parses_csv_with_string_dtype(s3_backend):
    dataset_id = uuid.uuid4()
    storage_path = dataset_storage.save_dataset_bytes(dataset_id, b"col1,col2\n1,2\n")

    df = dataset_storage.read_dataset_dataframe(storage_path)

    assert list(df.columns) == ["col1", "col2"]
    assert df["col1"].iloc[0] == "1"  # dtype=str, same contract as load_raw_dataframe
    assert isinstance(df["col1"].iloc[0], str)


def test_read_dataset_dataframe_local_delegates_to_load_raw_dataframe(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("DATASET_STORAGE_BACKEND", raising=False)
    get_settings.cache_clear()

    dataset_id = uuid.uuid4()
    storage_path = dataset_storage.save_dataset_bytes(dataset_id, b"col1,col2\n1,2\n")

    df = dataset_storage.read_dataset_dataframe(storage_path)

    assert list(df.columns) == ["col1", "col2"]
    assert df["col1"].iloc[0] == "1"
    get_settings.cache_clear()
