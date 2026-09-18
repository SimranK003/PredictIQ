"""Dataset validation for ingestion.

Runs a series of independent checks against an uploaded CSV (already
loaded into a DataFrame) and produces a structured data-quality report.
The ingestion API rejects the dataset (HTTP 422) when any check marked
as a hard failure fails; soft issues are reported but don't block.
"""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ml.schema import DatasetSchema


class Severity(str, Enum):
    ERROR = "error"  # blocks ingestion
    WARNING = "warning"  # reported but does not block


@dataclass
class ValidationIssue:
    check: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    issues: list[ValidationIssue] = field(default_factory=list)
    column_null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_row_count: int = 0
    class_balance: dict[str, int] | None = None

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "is_valid": self.is_valid,
            "duplicate_row_count": self.duplicate_row_count,
            "column_null_counts": self.column_null_counts,
            "class_balance": self.class_balance,
            "issues": [
                {
                    "check": i.check,
                    "severity": i.severity.value,
                    "message": i.message,
                    "details": i.details,
                }
                for i in self.issues
            ],
        }


MIN_ROWS_REQUIRED = 50


def validate_dataset(df: pd.DataFrame, schema: DatasetSchema) -> DataQualityReport:
    """Run all checks and return a complete report.

    The dataframe is expected to be raw (as read from CSV, dtype=str-ish
    is fine) — checks handle coercion themselves where the schema allows it.
    """
    issues: list[ValidationIssue] = []

    n_rows, n_columns = df.shape

    _check_empty(df, issues)
    _check_schema_columns(df, schema, issues)

    # Remaining checks need the required columns to be present to be meaningful.
    missing_required = {c for c in schema.required_columns if c not in df.columns}
    if not missing_required:
        _check_row_count(df, issues)
        _check_duplicates(df, schema, issues)
        null_counts = _check_missing_values(df, schema, issues)
        _check_dtypes(df, schema, issues)
        _check_target_labels(df, schema, issues)
        class_balance = _class_balance(df, schema)
    else:
        null_counts = {}
        class_balance = None

    if df.empty:
        duplicate_count = 0
    elif schema.id_column in df.columns:
        feature_cols = [c for c in df.columns if c != schema.id_column]
        duplicate_count = int(df.duplicated(subset=feature_cols).sum())
    else:
        duplicate_count = int(df.duplicated().sum())

    return DataQualityReport(
        n_rows=n_rows,
        n_columns=n_columns,
        issues=issues,
        column_null_counts=null_counts,
        duplicate_row_count=duplicate_count,
        class_balance=class_balance,
    )


def _check_empty(df: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if df.empty:
        issues.append(
            ValidationIssue(
                check="non_empty",
                severity=Severity.ERROR,
                message="Uploaded dataset has zero rows.",
            )
        )


def _check_schema_columns(
    df: pd.DataFrame, schema: DatasetSchema, issues: list[ValidationIssue]
) -> None:
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        issues.append(
            ValidationIssue(
                check="required_columns",
                severity=Severity.ERROR,
                message=f"Dataset is missing {len(missing)} required column(s).",
                details={"missing_columns": missing},
            )
        )

    extra = [c for c in df.columns if c not in schema.required_columns]
    if extra:
        issues.append(
            ValidationIssue(
                check="unexpected_columns",
                severity=Severity.WARNING,
                message=(
                    "Dataset contains columns not in the expected schema; they will be ignored."
                ),
                details={"extra_columns": extra},
            )
        )


def _check_row_count(df: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if len(df) < MIN_ROWS_REQUIRED:
        issues.append(
            ValidationIssue(
                check="minimum_row_count",
                severity=Severity.ERROR,
                message=(
                    f"Dataset has {len(df)} rows; at least {MIN_ROWS_REQUIRED} "
                    "are required for training."
                ),
                details={"row_count": len(df), "minimum_required": MIN_ROWS_REQUIRED},
            )
        )


def _check_duplicates(
    df: pd.DataFrame, schema: DatasetSchema, issues: list[ValidationIssue]
) -> None:
    id_dupes = int(df[schema.id_column].duplicated().sum())
    if id_dupes > 0:
        issues.append(
            ValidationIssue(
                check="duplicate_ids",
                severity=Severity.ERROR,
                message=f"Found {id_dupes} duplicate value(s) in id column '{schema.id_column}'.",
                details={"duplicate_id_count": id_dupes},
            )
        )

    # Duplicate *content*, ignoring the id column — two distinct customer
    # IDs with identical feature values still indicate a likely data issue
    # (e.g. an export bug), even though df.duplicated() on all columns
    # would never fire since ids are unique by definition.
    feature_cols = [c for c in df.columns if c != schema.id_column]
    content_dupes = int(df.duplicated(subset=feature_cols).sum())
    if content_dupes > 0:
        issues.append(
            ValidationIssue(
                check="duplicate_rows",
                severity=Severity.WARNING,
                message=(
                    f"Found {content_dupes} row(s) with duplicate feature values (different ids)."
                ),
                details={"duplicate_row_count": content_dupes},
            )
        )


def _check_missing_values(
    df: pd.DataFrame, schema: DatasetSchema, issues: list[ValidationIssue]
) -> dict[str, int]:
    null_counts: dict[str, int] = {}
    for col in schema.all_feature_columns:
        # Blank strings count as missing too, not just NaN.
        series = df[col]
        blank_mask = series.isna() | (series.astype(str).str.strip() == "")
        count = int(blank_mask.sum())
        null_counts[col] = count

        if count == 0:
            continue

        if col in schema.coercible_numeric_columns:
            issues.append(
                ValidationIssue(
                    check="missing_values",
                    severity=Severity.WARNING,
                    message=(
                        f"Column '{col}' has {count} missing/blank value(s); "
                        "will be imputed during preprocessing."
                    ),
                    details={"column": col, "missing_count": count},
                )
            )
        else:
            pct = count / len(df)
            severity = Severity.ERROR if pct > 0.5 else Severity.WARNING
            issues.append(
                ValidationIssue(
                    check="missing_values",
                    severity=severity,
                    message=f"Column '{col}' has {count} missing/blank value(s) ({pct:.1%}).",
                    details={
                        "column": col,
                        "missing_count": count,
                        "missing_fraction": round(pct, 4),
                    },
                )
            )

    target_nulls = int(df[schema.target_column].isna().sum())
    if target_nulls > 0:
        issues.append(
            ValidationIssue(
                check="missing_target",
                severity=Severity.ERROR,
                message=(
                    f"Target column '{schema.target_column}' has {target_nulls} missing value(s)."
                ),
                details={"missing_count": target_nulls},
            )
        )

    return null_counts


def _check_dtypes(df: pd.DataFrame, schema: DatasetSchema, issues: list[ValidationIssue]) -> None:
    for col in schema.numeric_columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        already_null = df[col].isna() | (df[col].astype(str).str.strip() == "")
        newly_invalid = coerced.isna() & ~already_null

        n_invalid = int(newly_invalid.sum())
        if n_invalid == 0:
            continue

        if col in schema.coercible_numeric_columns:
            issues.append(
                ValidationIssue(
                    check="dtype_numeric",
                    severity=Severity.WARNING,
                    message=(
                        f"Column '{col}' has {n_invalid} non-numeric value(s); "
                        "will be coerced/imputed."
                    ),
                    details={"column": col, "invalid_count": n_invalid},
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    check="dtype_numeric",
                    severity=Severity.ERROR,
                    message=(
                        f"Column '{col}' expected numeric values but found {n_invalid} that aren't."
                    ),
                    details={"column": col, "invalid_count": n_invalid},
                )
            )


def _check_target_labels(
    df: pd.DataFrame, schema: DatasetSchema, issues: list[ValidationIssue]
) -> None:
    valid_labels = {schema.positive_label, schema.negative_label}
    observed = set(df[schema.target_column].dropna().unique().tolist())

    if not observed.issubset(valid_labels):
        unexpected = observed - valid_labels
        issues.append(
            ValidationIssue(
                check="target_labels",
                severity=Severity.ERROR,
                message=f"Target column '{schema.target_column}' has unexpected label(s).",
                details={"unexpected_labels": sorted(unexpected)},
            )
        )
    elif len(observed) < 2:
        issues.append(
            ValidationIssue(
                check="target_labels",
                severity=Severity.ERROR,
                message=(
                    f"Target column '{schema.target_column}' must have at least 2 distinct "
                    f"classes; found {len(observed)}."
                ),
                details={"observed_labels": sorted(observed)},
            )
        )


def _class_balance(df: pd.DataFrame, schema: DatasetSchema) -> dict[str, int]:
    return df[schema.target_column].value_counts(dropna=True).to_dict()
