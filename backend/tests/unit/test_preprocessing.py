import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import ChurnFeatureEngineer, build_full_pipeline, build_preprocessing_pipeline
from ml.schema import CHURN_SCHEMA
from ml.training.config import TrainingConfig
from ml.training.models import build_logistic_regression


def _raw_rows(n: int = 20) -> pd.DataFrame:
    rows = {
        "tenure": [i % 10 for i in range(n)],
        "MonthlyCharges": [50.0 + i for i in range(n)],
        "TotalCharges": [str((50.0 + i) * (i % 10)) for i in range(n)],
        "gender": ["Male" if i % 2 == 0 else "Female" for i in range(n)],
        "SeniorCitizen": ["0"] * n,
        "Partner": ["Yes"] * n,
        "Dependents": ["No"] * n,
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
    }
    return pd.DataFrame(rows)


def test_feature_engineer_derives_avg_monthly_spend_for_existing_customers():
    df = _raw_rows(5)
    df["tenure"] = df["tenure"].astype(object)
    df.loc[0, "tenure"] = "10"
    df.loc[0, "TotalCharges"] = "500.0"
    out = ChurnFeatureEngineer().fit_transform(df)
    assert out.loc[0, "AvgMonthlySpend"] == pytest.approx(50.0)


def test_feature_engineer_falls_back_to_monthly_charges_when_tenure_zero():
    df = _raw_rows(5)
    df["tenure"] = df["tenure"].astype(object)
    df.loc[0, "tenure"] = "0"
    df.loc[0, "TotalCharges"] = ""
    df.loc[0, "MonthlyCharges"] = 75.0
    out = ChurnFeatureEngineer().fit_transform(df)
    assert out.loc[0, "AvgMonthlySpend"] == pytest.approx(75.0)


def test_feature_engineer_coerces_blank_total_charges_to_nan():
    df = _raw_rows(3)
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = ""
    out = ChurnFeatureEngineer().fit_transform(df)
    assert pd.isna(out.loc[0, "TotalCharges"])


def test_preprocessing_pipeline_produces_no_nans_after_transform():
    df = _raw_rows(30)
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = ""  # simulate the real dataset's blank-TotalCharges quirk

    engineered = ChurnFeatureEngineer().fit_transform(df)
    preprocessor = build_preprocessing_pipeline(CHURN_SCHEMA)
    transformed = preprocessor.fit_transform(engineered)

    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(dense).any()


def test_full_pipeline_is_fit_only_on_training_fold_not_full_data():
    """Leakage check: the fitted scaler's mean should reflect only the
    rows it was fit on, not statistics from data outside that fold.
    """
    df = _raw_rows(40)
    df["MonthlyCharges"] = [float(i) for i in range(40)]  # distinct, easy to reason about

    config = TrainingConfig()
    pipeline = build_full_pipeline(CHURN_SCHEMA, build_logistic_regression(config))

    train_df = df.iloc[:20]
    pipeline.fit(train_df, pd.Series([0, 1] * 10))

    numeric_transformer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
    scaler = numeric_transformer.named_steps["scaler"]
    numeric_features = list(CHURN_SCHEMA.numeric_columns) + ["AvgMonthlySpend"]
    monthly_charges_idx = numeric_features.index("MonthlyCharges")

    expected_mean = train_df["MonthlyCharges"].astype(float).mean()
    assert scaler.mean_[monthly_charges_idx] == pytest.approx(expected_mean)

    full_mean = df["MonthlyCharges"].astype(float).mean()
    assert scaler.mean_[monthly_charges_idx] != pytest.approx(full_mean)


def test_full_pipeline_predicts_without_error_on_held_out_rows():
    df = _raw_rows(40)
    y = pd.Series([0, 1] * 20)
    config = TrainingConfig()
    pipeline = build_full_pipeline(CHURN_SCHEMA, build_logistic_regression(config))

    pipeline.fit(df.iloc[:30], y.iloc[:30])
    proba = pipeline.predict_proba(df.iloc[30:])[:, 1]

    assert len(proba) == 10
    assert ((proba >= 0) & (proba <= 1)).all()
