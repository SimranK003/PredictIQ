"""Integration test for MLflow run logging.

Uses a local file-store tracking URI (no live MLflow server required) so
this test has no external service dependency — the same mlflow.* client
calls behave identically against a real tracking server, which is what
Docker Compose runs in production.
"""

import mlflow
import pandas as pd
import pytest

from ml.schema import CHURN_SCHEMA
from ml.training.config import TrainingConfig
from ml.training.train import log_training_run_to_mlflow, run_training_pipeline
from tests.integration.test_training_pipeline import _synthetic_churn_dataframe


class _FakeDataset:
    def __init__(self):
        self.id = "11111111-1111-1111-1111-111111111111"
        self.content_hash = "deadbeef" * 8
        self.filename = "synthetic.csv"


@pytest.fixture()
def mlflow_tmp_tracking(tmp_path, monkeypatch):
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("predictiq-test")
    yield tracking_uri


def test_log_training_run_round_trips_through_mlflow(mlflow_tmp_tracking):
    df = _synthetic_churn_dataframe(n=150, seed=3)
    config = TrainingConfig(random_forest_n_estimators=20, xgboost_n_estimators=20)
    results = run_training_pipeline(df, CHURN_SCHEMA, config)
    result = results[0]

    dataset = _FakeDataset()
    sample_input = df[list(CHURN_SCHEMA.all_feature_columns)].head(2)

    run_handle = log_training_run_to_mlflow(
        result, dataset=dataset, schema=CHURN_SCHEMA, config=config, sample_input=sample_input
    )

    run = mlflow.get_run(run_handle.run_id)
    assert run.data.tags["algorithm"] == result.algorithm
    assert run.data.tags["dataset_id"] == dataset.id
    assert run.data.tags["dataset_content_hash"] == dataset.content_hash
    assert "git_commit" in run.data.tags

    assert run.data.metrics["test_roc_auc"] == pytest.approx(result.test_metrics["roc_auc"])
    assert run.data.metrics["val_f1"] == pytest.approx(result.val_metrics["f1"])

    assert run_handle.artifact_uri == f"runs:/{run_handle.run_id}/model"
    loaded_model = mlflow.sklearn.load_model(run_handle.artifact_uri)
    reloaded_proba = loaded_model.predict_proba(sample_input)[:, 1]
    original_proba = result.pipeline.predict_proba(sample_input)[:, 1]

    pd.testing.assert_series_equal(
        pd.Series(reloaded_proba), pd.Series(original_proba), check_names=False
    )
