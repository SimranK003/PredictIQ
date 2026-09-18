"""Pydantic schemas for the inference API.

ChurnFeaturesIn is hand-written (not generated via pydantic.create_model)
so it can express per-field nuance — TotalCharges optionality mirrors
ml.schema.CHURN_SCHEMA.coercible_numeric_columns, and field descriptions
make the OpenAPI docs useful. To guarantee it can't silently drift from
the training schema, tests/unit/test_prediction_schema.py asserts its
field set exactly matches CHURN_SCHEMA.all_feature_columns — if someone
adds/removes a column in ml/schema.py without updating this model, that
test fails loudly instead of the two silently diverging.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from db.models import JobStatus


class ChurnFeaturesIn(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: str = Field(
        ..., examples=["0"], description="'0' or '1' — matches the raw dataset's encoding."
    )
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[2])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[95.0])
    TotalCharges: float | None = Field(
        None,
        ge=0,
        description="Omit for brand-new customers (tenure=0) with no bill yet; imputed downstream.",
        examples=[190.0],
    )


class PredictionOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction_id: uuid.UUID
    prediction: str
    probability: float
    model_version: str
    model_version_id: uuid.UUID
    timestamp: datetime
    request_id: uuid.UUID


class BatchPredictionRequest(BaseModel):
    records: list[ChurnFeaturesIn] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    job_id: uuid.UUID
    status: JobStatus
    n_records: int
    model_version: str


class PredictionSummaryOut(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    id: uuid.UUID
    request_id: uuid.UUID
    model_version_id: uuid.UUID
    prediction: str
    probability: float
    created_at: datetime


class PredictionDetailOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: uuid.UUID
    request_id: uuid.UUID
    model_version_id: uuid.UUID
    model_version_label: str
    algorithm: str
    input_features: dict
    prediction: str
    probability: float
    latency_ms: float
    created_at: datetime


class PaginatedPredictionsOut(BaseModel):
    items: list[PredictionSummaryOut]
    total: int
    limit: int
    offset: int
