"""Integration tests for GET /monitoring/drift against real Postgres and
a real training dataset file on disk (build_reference_distribution
re-reads it) — the ML pipeline itself isn't involved here, only the
statistics layer operating on real stored predictions.

DRIFT_MIN_SAMPLES is lowered via env var in most tests to keep fixture
data generation fast; the dedicated threshold tests set it explicitly
to make the boundary unambiguous rather than relying on the default.
"""

import random
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import get_settings
from db.models import ModelStage


@pytest.fixture()
def low_drift_threshold(monkeypatch):
    monkeypatch.setenv("DRIFT_MIN_SAMPLES", "20")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_drift_returns_no_production_model_when_none_registered(client):
    resp = client.get("/monitoring/drift")
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_production_model"}


def test_drift_returns_insufficient_data_below_configured_minimum(
    client, make_real_dataset_file, make_model_version, make_predictions, monkeypatch
):
    monkeypatch.setenv("DRIFT_MIN_SAMPLES", "50")
    get_settings.cache_clear()
    dataset = make_real_dataset_file()
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)
    make_predictions(model_version, 10)

    resp = client.get("/monitoring/drift")
    get_settings.cache_clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "insufficient_data"
    assert body["sample_size"] == 10
    assert body["minimum_required"] == 50


def test_drift_returns_ok_status_once_minimum_is_reached(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    dataset = make_real_dataset_file()
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)
    make_predictions(model_version, 25)

    resp = client.get("/monitoring/drift")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["sample_size"] == 25
    assert body["reference_dataset_id"] == str(dataset.id)
    assert body["reference_dataset_content_hash"] == dataset.content_hash
    assert len(body["features"]) == 19  # all CHURN_SCHEMA feature columns


def test_drift_no_drift_when_production_matches_training_distribution(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    """Production tenure values drawn from the same range as training
    (uniform 0-72, matching _synthetic_churn_dataframe) should not trip
    the KS threshold.
    """
    dataset = make_real_dataset_file(n_rows=300)
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)

    rng = random.Random(99)

    def _matching_features(i):
        from tests.conftest import sample_churn_features

        return sample_churn_features(tenure=rng.randint(0, 72))

    make_predictions(model_version, 60, feature_fn=_matching_features)

    resp = client.get("/monitoring/drift", params={"feature": "tenure"})

    assert resp.status_code == 200
    body = resp.json()
    tenure_result = body["features"][0]
    assert tenure_result["feature"] == "tenure"
    assert tenure_result["test"] == "KS"
    assert tenure_result["drift_detected"] is False


def test_drift_detects_a_deliberate_tenure_shift(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    """Production tenure values concentrated at the extreme high end
    (70-72) versus training's uniform 0-72 spread is a real, deliberate
    shift that KS should catch.
    """
    dataset = make_real_dataset_file(n_rows=300)
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)

    def _shifted_features(i):
        from tests.conftest import sample_churn_features

        return sample_churn_features(tenure=70 + (i % 3))

    make_predictions(model_version, 60, feature_fn=_shifted_features)

    resp = client.get("/monitoring/drift", params={"feature": "tenure"})

    assert resp.status_code == 200
    body = resp.json()
    tenure_result = body["features"][0]
    assert tenure_result["drift_detected"] is True
    assert tenure_result["score"] >= tenure_result["threshold"]
    assert body["drift_detected"] is True


def test_drift_detects_a_categorical_proportion_shift(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    """Training data's Contract categories are split across three
    values; production traffic that's become 100% Month-to-month is a
    real proportion shift PSI should catch.
    """
    dataset = make_real_dataset_file(n_rows=300)
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)

    def _all_month_to_month(i):
        from tests.conftest import sample_churn_features

        return sample_churn_features(Contract="Month-to-month")

    make_predictions(model_version, 60, feature_fn=_all_month_to_month)

    resp = client.get("/monitoring/drift", params={"feature": "Contract"})

    assert resp.status_code == 200
    body = resp.json()
    contract_result = body["features"][0]
    assert contract_result["feature"] == "Contract"
    assert contract_result["test"] == "PSI"
    assert contract_result["drift_detected"] is True
    assert contract_result["severity"] in {"warning", "critical"}


def test_drift_model_version_isolation(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    """Predictions from a different (e.g. previous) model version must
    never leak into another model version's drift sample.
    """
    dataset_a = make_real_dataset_file()
    dataset_b = make_real_dataset_file()
    model_a = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset_a)
    model_b = make_model_version(stage=ModelStage.PREVIOUS, dataset=dataset_b)

    make_predictions(model_a, 25)
    make_predictions(model_b, 40)

    resp_a = client.get("/monitoring/drift", params={"model_version_id": str(model_a.id)})
    resp_b = client.get("/monitoring/drift", params={"model_version_id": str(model_b.id)})

    assert resp_a.json()["sample_size"] == 25
    assert resp_b.json()["sample_size"] == 40
    assert resp_a.json()["reference_dataset_id"] == str(dataset_a.id)
    assert resp_b.json()["reference_dataset_id"] == str(dataset_b.id)


def test_drift_time_window_filtering(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    dataset = make_real_dataset_file()
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)

    old_timestamp = datetime.now(UTC) - timedelta(days=10)
    make_predictions(model_version, 30, created_at=old_timestamp)
    make_predictions(model_version, 25)  # recent, default created_at=now()

    mv_id = str(model_version.id)
    resp_all_time = client.get("/monitoring/drift", params={"model_version_id": mv_id})
    resp_24h = client.get(
        "/monitoring/drift", params={"model_version_id": mv_id, "window": "24h"}
    )

    assert resp_all_time.json()["sample_size"] == 55
    assert resp_24h.json()["sample_size"] == 25


def test_drift_returns_404_for_unknown_model_version(client):
    resp = client.get(
        "/monitoring/drift", params={"model_version_id": "00000000-0000-0000-0000-000000000000"}
    )
    assert resp.status_code == 404


def test_drift_returns_400_for_unknown_feature(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    dataset = make_real_dataset_file()
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)
    make_predictions(model_version, 25)

    resp = client.get(
        "/monitoring/drift",
        params={"model_version_id": str(model_version.id), "feature": "not_a_real_feature"},
    )
    assert resp.status_code == 400


def test_drift_returns_400_for_unknown_window(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    dataset = make_real_dataset_file()
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)
    make_predictions(model_version, 25)

    resp = client.get(
        "/monitoring/drift",
        params={"model_version_id": str(model_version.id), "window": "3_fortnights"},
    )
    assert resp.status_code == 400


def test_drift_response_never_includes_raw_input_features(
    client, make_real_dataset_file, make_model_version, make_predictions, low_drift_threshold
):
    """Monitoring must return aggregates only — feature *names* like
    "PaymentMethod" are expected (they label which aggregate a result is
    for), but no raw per-prediction value like "Electronic check" or the
    input_features key itself should ever appear.
    """
    dataset = make_real_dataset_file()
    model_version = make_model_version(stage=ModelStage.PRODUCTION, dataset=dataset)
    make_predictions(model_version, 25)

    resp = client.get("/monitoring/drift", params={"model_version_id": str(model_version.id)})

    body_text = resp.text
    assert "input_features" not in body_text
    assert "Electronic check" not in body_text
    feature_names = {f["feature"] for f in resp.json()["features"]}
    assert "PaymentMethod" in feature_names
