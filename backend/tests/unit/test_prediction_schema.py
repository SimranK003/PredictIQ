"""Guards against ChurnFeaturesIn silently drifting from the training
schema — if someone adds/removes a column in ml/schema.py without
updating the hand-written Pydantic model (or vice versa), this fails
loudly instead of the two quietly diverging.
"""

from app.schemas.prediction import ChurnFeaturesIn
from ml.schema import CHURN_SCHEMA


def test_prediction_schema_fields_match_training_feature_columns():
    schema_fields = set(ChurnFeaturesIn.model_fields.keys())
    expected = set(CHURN_SCHEMA.all_feature_columns)
    assert schema_fields == expected


def test_total_charges_is_optional_matching_coercible_numeric_columns():
    """TotalCharges is the one column ml/schema.py marks coercible
    (blank for brand-new customers) — the API contract should mirror
    that, not require a value the trained pipeline already knows how to
    impute.
    """
    field = ChurnFeaturesIn.model_fields["TotalCharges"]
    assert field.is_required() is False
    for col in CHURN_SCHEMA.all_feature_columns:
        if col not in CHURN_SCHEMA.coercible_numeric_columns:
            assert ChurnFeaturesIn.model_fields[col].is_required() is True, col
