"""Pydantic response models for the model registry API."""

import uuid
from datetime import datetime

from pydantic import BaseModel

from db.models import ModelStage


class ModelVersionOut(BaseModel):
    id: uuid.UUID
    version_label: str
    algorithm: str
    stage: ModelStage
    mlflow_run_id: str
    mlflow_experiment_id: str
    artifact_uri: str
    git_commit: str
    metrics: dict
    params: dict
    dataset_id: uuid.UUID
    created_at: datetime
    promoted_at: datetime | None

    model_config = {"from_attributes": True}


class ModelVersionSummaryOut(BaseModel):
    id: uuid.UUID
    version_label: str
    algorithm: str
    stage: ModelStage
    metrics: dict
    created_at: datetime
    promoted_at: datetime | None

    model_config = {"from_attributes": True}


class ModelMetricsOut(BaseModel):
    id: uuid.UUID
    version_label: str
    algorithm: str
    metrics: dict

    model_config = {"from_attributes": True}


class PromoteRequest(BaseModel):
    candidate_id: uuid.UUID


class LifecycleEventOut(BaseModel):
    action: str
    previous_stage: ModelStage | None
    new_stage: ModelStage
    triggered_by: str
    event_metadata: dict
    created_at: datetime

    model_config = {"from_attributes": True}
