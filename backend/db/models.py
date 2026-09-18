"""SQLAlchemy ORM models.

Covers the full data model for the platform (datasets, jobs, model
registry, predictions). Only the `datasets` table is exercised by Phase 1
(ingestion); the rest are defined now so the schema is coherent and
Alembic migrations don't need to be revisited piecemeal, but they're
populated by later phases (training, inference, monitoring).
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobType(str, enum.Enum):
    INGEST = "ingest"
    TRAIN = "train"
    BATCH_PREDICT = "batch_predict"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStage(str, enum.Enum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    PREVIOUS = "previous"
    ARCHIVED = "archived"


class LifecycleAction(str, enum.Enum):
    REGISTERED = "registered"
    PROMOTED = "promoted"
    DEMOTED_TO_PREVIOUS = "demoted_to_previous"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    schema_name: Mapped[str] = mapped_column(String(100), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_columns: Mapped[int] = mapped_column(Integer, nullable=False)
    is_valid: Mapped[bool] = mapped_column(nullable=False)
    quality_report: Mapped[dict] = mapped_column(JSONB, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    model_versions: Mapped[list["ModelVersionRecord"]] = relationship(back_populates="dataset")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type: Mapped[JobType] = mapped_column(Enum(JobType, name="job_type"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), nullable=False, default=JobStatus.QUEUED
    )
    # Nullable: only TRAIN (and, in principle, a future async INGEST) jobs
    # have a triggering dataset; BATCH_PREDICT jobs pin a model version
    # instead (still tracked via payload — see app/routers/predictions.py).
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("datasets.id"), nullable=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(155), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped["Dataset | None"] = relationship()
    model_versions: Mapped[list["ModelVersionRecord"]] = relationship(
        back_populates="training_job"
    )


class ModelVersionRecord(Base):
    """Our own model registry table — the source of truth for the
    candidate/production/previous promotion workflow. MLflow remains the
    source of truth for run params/metrics/artifacts; this table just
    references an MLflow run and tracks its business lifecycle stage.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        # DB-enforced invariants, not just application logic: at most one
        # row can hold each of these two stages at any time. A partial
        # unique index on a constant-per-row-subset column works because
        # every row matching the predicate has the same indexed value, so
        # a second matching row would collide.
        # NOTE: SQLAlchemy's Enum type stores the Python member *name*
        # (e.g. "PRODUCTION"), not its .value ("production"), as the
        # actual Postgres enum label — these predicates must match that.
        Index(
            "uq_model_versions_single_production",
            "stage",
            unique=True,
            postgresql_where=text("stage = 'PRODUCTION'"),
        ),
        Index(
            "uq_model_versions_single_previous",
            "stage",
            unique=True,
            postgresql_where=text("stage = 'PREVIOUS'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_label: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    mlflow_experiment_id: Mapped[str] = mapped_column(String(100), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[ModelStage] = mapped_column(
        Enum(ModelStage, name="model_stage"), nullable=False, default=ModelStage.CANDIDATE
    )
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    # Nullable: candidates registered by the original CLI training path
    # (Phase 2, before async training jobs existed) have no Job row —
    # that's an honest gap, not something to backfill with a fake job.
    training_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="model_versions")
    training_job: Mapped["Job | None"] = relationship(back_populates="model_versions")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")
    lifecycle_events: Mapped[list["ModelLifecycleEvent"]] = relationship(
        back_populates="model_version", order_by="ModelLifecycleEvent.created_at"
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        # Common query patterns: "predictions for model X, newest first"
        # (dashboard/history filtering) and "predictions in a date range"
        # (drift analysis, debugging a specific time window).
        Index("ix_predictions_model_version_created_at", "model_version_id", "created_at"),
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_request_id", "request_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, default=uuid.uuid4
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    input_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    prediction: Mapped[str] = mapped_column(String(50), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model_version: Mapped["ModelVersionRecord"] = relationship(back_populates="predictions")


class ModelLifecycleEvent(Base):
    """Audit trail for model registry lifecycle transitions.

    Deliberately minimal: no actor/user tracking since there's no auth
    layer yet (`triggered_by` records "system" for training-time
    registration or "api" for promote/rollback calls). This is enough to
    answer "what happened to this model version and when," which is what
    an application-level audit trail needs to do.
    """

    __tablename__ = "model_lifecycle_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    action: Mapped[LifecycleAction] = mapped_column(
        Enum(LifecycleAction, name="lifecycle_action"), nullable=False
    )
    previous_stage: Mapped[ModelStage | None] = mapped_column(
        Enum(ModelStage, name="model_stage"), nullable=True
    )
    new_stage: Mapped[ModelStage] = mapped_column(
        Enum(ModelStage, name="model_stage"), nullable=False
    )
    triggered_by: Mapped[str] = mapped_column(String(50), nullable=False, default="api")
    event_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model_version: Mapped["ModelVersionRecord"] = relationship(back_populates="lifecycle_events")
