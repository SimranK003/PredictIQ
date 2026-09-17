"""Dataset schema contract for the churn use case.

This module is the single source of truth for what a valid input dataset
looks like. Ingestion validation, preprocessing, and training all import
from here instead of re-declaring column lists, so the schema can't drift
between the three stages.

Designed so a future tabular classification use case only needs a new
DatasetSchema instance, not changes to the ingestion/preprocessing/training
code paths themselves.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSchema:
    """Describes the shape of a tabular classification dataset."""

    name: str
    id_column: str
    target_column: str
    positive_label: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    # Columns that are legitimately allowed to be missing/blank and should
    # be coerced rather than rejected outright (e.g. "TotalCharges" has
    # blank strings for brand-new customers with tenure == 0).
    coercible_numeric_columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.id_column, *self.all_feature_columns, self.target_column)


CHURN_SCHEMA = DatasetSchema(
    name="telco_customer_churn",
    id_column="customerID",
    target_column="Churn",
    positive_label="Yes",
    numeric_columns=(
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ),
    categorical_columns=(
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ),
    coercible_numeric_columns=("TotalCharges",),
)
