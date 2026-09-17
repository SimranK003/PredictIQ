import io

import pandas as pd

from ml.schema import CHURN_SCHEMA


def _churn_csv_bytes(n: int = 60, corrupt: bool = False) -> bytes:
    data = {
        "customerID": [f"C{i}" for i in range(n)],
        "gender": ["Male" if i % 2 == 0 else "Female" for i in range(n)],
        "SeniorCitizen": [0] * n,
        "Partner": ["Yes" if i % 3 == 0 else "No" for i in range(n)],
        "Dependents": ["No"] * n,
        "tenure": list(range(n)),
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No"] * n,
        "InternetService": ["DSL"] * n,
        "OnlineSecurity": ["No"] * n,
        "OnlineBackup": ["No"] * n,
        "DeviceProtection": ["No"] * n,
        "TechSupport": ["No"] * n,
        "StreamingTV": ["No"] * n,
        "StreamingMovies": ["No"] * n,
        "Contract": ["Month-to-month"] * n,
        "PaperlessBilling": ["Yes"] * n,
        "PaymentMethod": ["Electronic check"] * n,
        "MonthlyCharges": [50.0 + i for i in range(n)],
        "TotalCharges": [str(50.0 * i) for i in range(n)],
        "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
    }
    df = pd.DataFrame(data)
    if corrupt:
        df = df.drop(columns=["tenure"])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_upload_valid_dataset_returns_201_and_report(client):
    files = {"file": ("churn.csv", _churn_csv_bytes(), "text/csv")}
    resp = client.post("/datasets", files=files)

    assert resp.status_code == 201
    body = resp.json()
    assert body["is_valid"] is True
    assert body["n_rows"] == 60
    assert body["schema_name"] == CHURN_SCHEMA.name
    assert body["quality_report"]["issues"] == []


def test_upload_invalid_dataset_returns_422_with_report(client):
    files = {"file": ("bad.csv", _churn_csv_bytes(corrupt=True), "text/csv")}
    resp = client.post("/datasets", files=files)

    assert resp.status_code == 422
    body = resp.json()
    issues = body["detail"]["quality_report"]["issues"]
    assert any(i["check"] == "required_columns" for i in issues)


def test_upload_non_csv_file_returns_400(client):
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    resp = client.post("/datasets", files=files)
    assert resp.status_code == 400


def test_upload_empty_file_returns_400(client):
    files = {"file": ("empty.csv", b"", "text/csv")}
    resp = client.post("/datasets", files=files)
    assert resp.status_code == 400


def test_get_dataset_by_id_roundtrips(client):
    files = {"file": ("churn.csv", _churn_csv_bytes(), "text/csv")}
    created = client.post("/datasets", files=files).json()

    resp = client.get(f"/datasets/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


def test_get_nonexistent_dataset_returns_404(client):
    resp = client.get("/datasets/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_list_datasets_includes_uploaded_ones(client):
    files = {"file": ("churn.csv", _churn_csv_bytes(), "text/csv")}
    created = client.post("/datasets", files=files).json()

    resp = client.get("/datasets")
    assert resp.status_code == 200
    ids = [d["id"] for d in resp.json()]
    assert created["id"] in ids


def test_health_endpoint_reports_database_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "ok"}
