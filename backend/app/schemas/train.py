"""Pydantic schemas for the training API.

TrainRequest's optional hyperparameter fields are hand-written (not
dynamically generated) so the OpenAPI docs are discoverable, but every
field name mirrors ml.training.config.TrainingConfig exactly — a test
(tests/unit/test_train_schema.py) asserts that correspondence so the two
can't silently drift apart.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from db.models import JobStatus
from ml.training.models import ALGORITHMS


class TrainRequest(BaseModel):
    dataset_id: uuid.UUID
    algorithms: list[str] | None = Field(
        None,
        description="Subset of algorithms to train; omit to train all of: " + ", ".join(ALGORITHMS),
    )

    # Split
    test_size: float | None = Field(None, gt=0, lt=1)
    val_size: float | None = Field(None, gt=0, lt=1)
    random_state: int | None = None

    # Logistic Regression
    logistic_regression_c: float | None = Field(None, gt=0)
    logistic_regression_max_iter: int | None = Field(None, gt=0)

    # Random Forest
    random_forest_n_estimators: int | None = Field(None, gt=0)
    random_forest_max_depth: int | None = Field(None, gt=0)
    random_forest_min_samples_leaf: int | None = Field(None, gt=0)

    # XGBoost
    xgboost_n_estimators: int | None = Field(None, gt=0)
    xgboost_max_depth: int | None = Field(None, gt=0)
    xgboost_learning_rate: float | None = Field(None, gt=0)
    xgboost_subsample: float | None = Field(None, gt=0, le=1)

    @field_validator("algorithms")
    @classmethod
    def _validate_algorithm_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("algorithms, if provided, must not be an empty list.")
        unknown = sorted(set(value) - set(ALGORITHMS))
        if unknown:
            raise ValueError(f"Unknown algorithm(s): {unknown}. Valid: {list(ALGORITHMS)}.")
        return value

    def resolved_config_overrides(self) -> dict:
        """Only the fields the caller actually set, suitable for
        TrainingConfig(**overrides) — omitted fields keep their default.
        """
        exclude = {"dataset_id", "algorithms"}
        return {
            k: v
            for k, v in self.model_dump().items()
            if k not in exclude and v is not None
        }


class TrainResponse(BaseModel):
    job_id: uuid.UUID
    dataset_id: uuid.UUID
    status: JobStatus
    created_at: datetime
