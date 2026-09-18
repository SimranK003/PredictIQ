"""Verifies default_artifact_exists against a real (local file-store)
MLflow — not a stub — so the "candidate must have a valid artifact"
policy check is proven against actual MLflow behavior, not an assumption
about its API. Uses a file-store tracking URI so it needs no live server.
"""

import mlflow

from app.services.registry import default_artifact_exists
from ml.schema import CHURN_SCHEMA
from ml.training.config import TrainingConfig
from ml.training.train import log_training_run_to_mlflow, run_training_pipeline
from tests.integration.test_training_pipeline import _synthetic_churn_dataframe


class _FakeDataset:
    def __init__(self):
        self.id = "22222222-2222-2222-2222-222222222222"
        self.content_hash = "cafebabe" * 8
        self.filename = "synthetic.csv"


def test_default_artifact_exists_true_for_a_real_logged_model(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    mlflow.set_experiment("predictiq-artifact-check-test")

    df = _synthetic_churn_dataframe(n=120, seed=11)
    config = TrainingConfig(random_forest_n_estimators=10, xgboost_n_estimators=10)
    result = run_training_pipeline(df, CHURN_SCHEMA, config)[0]
    sample_input = df[list(CHURN_SCHEMA.all_feature_columns)].head(2)

    run_handle = log_training_run_to_mlflow(
        result,
        dataset=_FakeDataset(),
        schema=CHURN_SCHEMA,
        config=config,
        sample_input=sample_input,
    )

    assert default_artifact_exists(run_handle.run_id) is True


def test_default_artifact_exists_false_for_a_nonexistent_run(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    assert default_artifact_exists("not-a-real-run-id") is False
