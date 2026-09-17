"""Shared preprocessing + feature engineering pipeline.

This is the single definition of "how raw feature columns become model
input." It is used to build the full sklearn Pipeline that gets fit
during training and logged to MLflow as one artifact. Inference (Phase 4)
loads that same fitted Pipeline object and calls .predict_proba() on raw
feature rows directly — there is no second, separately-maintained
transformation step at serving time to drift out of sync with training.

Only ChurnFeatureEngineer is schema-specific (it references named Telco
columns). build_preprocessing_pipeline itself is generic over any
DatasetSchema, so a future non-churn tabular use case only needs its own
small feature-engineering transformer, not a rewrite of this module.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.schema import DatasetSchema

ENGINEERED_NUMERIC_FEATURES = ("AvgMonthlySpend",)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic, row-wise feature derivation — no fitting required.

    - Coerces TotalCharges to numeric (blank strings, e.g. brand-new
      customers with tenure == 0, become NaN and are left for the
      downstream imputer to handle — imputation must happen inside the
      pipeline so it's fit on the training fold only).
    - Derives AvgMonthlySpend = TotalCharges / tenure. For tenure == 0
      customers this ratio is undefined, so it falls back to their
      current MonthlyCharges as the best available estimate of their
      spend rate. Both operations are pure functions of a row's own
      values, not of the target or of other rows, so applying this
      before the train/val/test split introduces no leakage.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        monthly_charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

        df["TotalCharges"] = total_charges
        df["AvgMonthlySpend"] = np.where(
            tenure > 0, total_charges / tenure.replace(0, np.nan), monthly_charges
        )
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + list(ENGINEERED_NUMERIC_FEATURES), dtype=object)


def build_preprocessing_pipeline(schema: DatasetSchema) -> ColumnTransformer:
    """Build the (unfit) ColumnTransformer for a schema's feature set.

    Numeric: median-impute (robust to the outlier-heavy MonthlyCharges/
    TotalCharges distributions) then standard-scale.
    Categorical: most-frequent-impute then one-hot encode, ignoring
    categories seen at inference time that weren't in training rather
    than raising — real production traffic will eventually include
    values training never saw.
    """
    numeric_features = list(schema.numeric_columns) + list(ENGINEERED_NUMERIC_FEATURES)
    categorical_features = list(schema.categorical_columns)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_full_pipeline(schema: DatasetSchema, classifier) -> Pipeline:
    """Feature engineering -> preprocessing -> classifier, as one Pipeline.

    This whole object is what gets .fit() during training and logged to
    MLflow — and what inference deserializes and calls directly.
    """
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessing_pipeline(schema)),
            ("classifier", classifier),
        ]
    )
