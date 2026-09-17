"""Integration tests for the pure training pipeline (no MLflow).

Uses a synthetic-but-schema-shaped dataset with an actual learnable
signal (low tenure + high monthly charges => churn) so ROC-AUC is
meaningfully above 0.5, not just "doesn't crash."
"""

import random

import pandas as pd
import pytest

from ml.schema import CHURN_SCHEMA
from ml.training.config import TrainingConfig
from ml.training.models import ALGORITHMS
from ml.training.train import run_training_pipeline


def _synthetic_churn_dataframe(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(0, 72)
        monthly_charges = round(rng.uniform(18.0, 120.0), 2)
        total_charges = "" if tenure == 0 else str(round(monthly_charges * tenure, 2))

        # Real signal: short tenure + high monthly charges strongly predicts churn,
        # with some noise so it's not trivially separable.
        churn_score = (1 if tenure < 12 else 0) + (1 if monthly_charges > 80 else 0)
        churn_prob = {0: 0.03, 1: 0.30, 2: 0.85}[churn_score]
        churn = "Yes" if rng.random() < churn_prob else "No"

        rows.append(
            {
                "customerID": f"SYN{i:04d}",
                "gender": rng.choice(["Male", "Female"]),
                "SeniorCitizen": rng.choice(["0", "1"]),
                "Partner": rng.choice(["Yes", "No"]),
                "Dependents": rng.choice(["Yes", "No"]),
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": rng.choice(["Yes", "No"]),
                "InternetService": rng.choice(["DSL", "Fiber optic", "No"]),
                "OnlineSecurity": rng.choice(["Yes", "No"]),
                "OnlineBackup": rng.choice(["Yes", "No"]),
                "DeviceProtection": rng.choice(["Yes", "No"]),
                "TechSupport": rng.choice(["Yes", "No"]),
                "StreamingTV": rng.choice(["Yes", "No"]),
                "StreamingMovies": rng.choice(["Yes", "No"]),
                "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
                "PaperlessBilling": rng.choice(["Yes", "No"]),
                "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer"]),
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Churn": churn,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _synthetic_churn_dataframe()


def test_all_algorithms_train_and_produce_valid_metrics(synthetic_df):
    config = TrainingConfig()
    results = run_training_pipeline(synthetic_df, CHURN_SCHEMA, config)

    assert {r.algorithm for r in results} == set(ALGORITHMS)
    for result in results:
        for metric_set in (result.val_metrics, result.test_metrics):
            assert 0.0 <= metric_set["precision"] <= 1.0
            assert 0.0 <= metric_set["recall"] <= 1.0
            assert 0.0 <= metric_set["f1"] <= 1.0
            assert 0.0 <= metric_set["roc_auc"] <= 1.0
            cm = metric_set["confusion_matrix"]
            total = sum(cm.values())
            assert total == result.n_test or total == result.n_val


def test_pipeline_learns_real_signal_better_than_random(synthetic_df):
    """With the engineered signal above, every model should beat random
    guessing (roc_auc > 0.5) by a comfortable margin on held-out test data.
    """
    config = TrainingConfig()
    results = run_training_pipeline(synthetic_df, CHURN_SCHEMA, config)

    for result in results:
        assert result.test_metrics["roc_auc"] > 0.6, (
            f"{result.algorithm} roc_auc={result.test_metrics['roc_auc']} did not beat random"
        )


def test_training_is_reproducible_given_same_random_state(synthetic_df):
    config = TrainingConfig()
    results_a = run_training_pipeline(synthetic_df, CHURN_SCHEMA, config)
    results_b = run_training_pipeline(synthetic_df, CHURN_SCHEMA, config)

    for a, b in zip(results_a, results_b, strict=True):
        assert a.test_metrics["roc_auc"] == pytest.approx(b.test_metrics["roc_auc"])
        assert a.test_metrics["f1"] == pytest.approx(b.test_metrics["f1"])


def test_split_sizes_respect_config_fractions(synthetic_df):
    config = TrainingConfig(test_size=0.2, val_size=0.2, random_state=1)
    results = run_training_pipeline(synthetic_df, CHURN_SCHEMA, config)

    n = len(synthetic_df)
    result = results[0]
    assert result.n_test == pytest.approx(n * 0.2, abs=2)
    assert result.n_val == pytest.approx(n * 0.2, abs=2)
    assert result.n_train == pytest.approx(n * 0.6, abs=2)
