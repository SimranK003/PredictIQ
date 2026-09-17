"""Dataset loading and train/val/test splitting for training.

Splitting happens on the raw dataframe, before any fitting occurs, so
that every downstream fitted step (imputers, scaler, encoder) only ever
sees the training fold — the standard sklearn-pipeline way to prevent
preprocessing leakage.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.schema import DatasetSchema


@dataclass
class DatasetSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def load_raw_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str)


def prepare_features_and_target(
    df: pd.DataFrame, schema: DatasetSchema
) -> tuple[pd.DataFrame, pd.Series]:
    x = df[list(schema.all_feature_columns)].copy()
    y = (df[schema.target_column] == schema.positive_label).astype(int)
    return x, y


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> DatasetSplit:
    """Stratified train/val/test split.

    val_size and test_size are both fractions of the *original* dataset;
    the train/val split is recomputed against the remainder so the final
    proportions match what was requested.
    """
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    remaining_val_fraction = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=remaining_val_fraction,
        random_state=random_state,
        stratify=y_train_val,
    )

    return DatasetSplit(
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test
    )
