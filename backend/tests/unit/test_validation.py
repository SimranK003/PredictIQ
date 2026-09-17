import pandas as pd
import pytest

from ml.schema import DatasetSchema
from ml.validation import Severity, validate_dataset

SCHEMA = DatasetSchema(
    name="test_schema",
    id_column="id",
    target_column="target",
    positive_label="Yes",
    numeric_columns=("age", "balance"),
    categorical_columns=("plan",),
    coercible_numeric_columns=("balance",),
)


def _find_issue(report, check: str, column: str | None = None):
    for issue in report.issues:
        if issue.check == check and (column is None or issue.details.get("column") == column):
            return issue
    return None


def _valid_rows(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"C{i}" for i in range(n)],
            "age": [30 + (i % 20) for i in range(n)],
            "balance": [100.0 + i for i in range(n)],
            "plan": ["basic" if i % 2 == 0 else "premium" for i in range(n)],
            "target": ["Yes" if i % 3 == 0 else "No" for i in range(n)],
        }
    )


def test_valid_dataset_passes():
    report = validate_dataset(_valid_rows(), SCHEMA)
    assert report.is_valid
    assert report.n_rows == 60
    assert report.duplicate_row_count == 0
    assert report.class_balance == {"No": 40, "Yes": 20}


def test_empty_dataset_is_rejected():
    report = validate_dataset(pd.DataFrame(), SCHEMA)
    assert not report.is_valid
    assert any(i.check == "non_empty" for i in report.issues)


def test_missing_required_column_is_rejected():
    df = _valid_rows().drop(columns=["age"])
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    col_issue = next(i for i in report.issues if i.check == "required_columns")
    assert "age" in col_issue.details["missing_columns"]


def test_unexpected_extra_column_is_warning_only():
    df = _valid_rows()
    df["extra_col"] = "noise"
    report = validate_dataset(df, SCHEMA)
    assert report.is_valid
    warning = next(i for i in report.issues if i.check == "unexpected_columns")
    assert warning.severity == Severity.WARNING


def test_below_minimum_row_count_is_rejected():
    report = validate_dataset(_valid_rows(10), SCHEMA)
    assert not report.is_valid
    assert any(i.check == "minimum_row_count" for i in report.issues)


def test_duplicate_ids_are_rejected():
    df = _valid_rows()
    df.loc[1, "id"] = df.loc[0, "id"]
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    dup_issue = next(i for i in report.issues if i.check == "duplicate_ids")
    assert dup_issue.details["duplicate_id_count"] == 1


def test_duplicate_feature_values_with_different_ids_is_warning_only():
    df = _valid_rows()
    cols = ["age", "balance", "plan", "target"]
    df.loc[1, cols] = df.loc[0, cols].to_numpy()
    report = validate_dataset(df, SCHEMA)
    warning = _find_issue(report, "duplicate_rows")
    assert warning is not None
    assert warning.severity == Severity.WARNING
    assert report.duplicate_row_count == 1


def test_missing_values_in_required_categorical_column_is_error_over_threshold():
    df = _valid_rows()
    df.loc[: len(df) // 2, "plan"] = None  # >50% missing
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    issue = _find_issue(report, "missing_values", "plan")
    assert issue.severity == Severity.ERROR


def test_missing_values_in_coercible_numeric_column_is_warning():
    df = _valid_rows()
    df["balance"] = df["balance"].astype(object)
    df.loc[0, "balance"] = ""
    report = validate_dataset(df, SCHEMA)
    assert report.is_valid
    issue = _find_issue(report, "missing_values", "balance")
    assert issue.severity == Severity.WARNING


def test_non_numeric_value_in_strict_numeric_column_is_rejected():
    df = _valid_rows()
    df["age"] = df["age"].astype(object)
    df.loc[0, "age"] = "not_a_number"
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    issue = _find_issue(report, "dtype_numeric", "age")
    assert issue.severity == Severity.ERROR


def test_non_numeric_value_in_coercible_column_is_warning():
    df = _valid_rows()
    df["balance"] = df["balance"].astype(object)
    df.loc[0, "balance"] = "N/A"
    report = validate_dataset(df, SCHEMA)
    assert report.is_valid
    issue = _find_issue(report, "dtype_numeric", "balance")
    assert issue.severity == Severity.WARNING


def test_unexpected_target_label_is_rejected():
    df = _valid_rows()
    df.loc[0, "target"] = "Maybe"
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    issue = next(i for i in report.issues if i.check == "target_labels")
    assert "Maybe" in issue.details["unexpected_labels"]


def test_single_class_target_is_rejected():
    df = _valid_rows()
    df["target"] = "Yes"
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    assert any(i.check == "target_labels" for i in report.issues)


def test_missing_target_values_are_rejected():
    df = _valid_rows()
    df.loc[0, "target"] = None
    report = validate_dataset(df, SCHEMA)
    assert not report.is_valid
    assert any(i.check == "missing_target" for i in report.issues)


@pytest.mark.parametrize("n_missing_cols", [1, 2])
def test_report_serializes_to_dict(n_missing_cols):
    df = _valid_rows()
    report = validate_dataset(df, SCHEMA)
    payload = report.to_dict()
    assert payload["is_valid"] is True
    assert payload["n_rows"] == 60
    assert isinstance(payload["issues"], list)
