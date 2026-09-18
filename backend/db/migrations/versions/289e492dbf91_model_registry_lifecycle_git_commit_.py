"""model registry lifecycle: git_commit, single-production/previous constraints, lifecycle events

Revision ID: 289e492dbf91
Revises: 19510ec9fab3
Create Date: 2026-09-18 10:07:04.805333

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '289e492dbf91'
down_revision: Union[str, None] = '19510ec9fab3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # model_stage already exists (created by the initial migration for
    # model_versions.stage) — reference it with create_type=False so this
    # migration doesn't try to CREATE TYPE a second time.
    model_stage = postgresql.ENUM(
        "CANDIDATE", "PRODUCTION", "PREVIOUS", "ARCHIVED", name="model_stage", create_type=False
    )

    op.create_table(
        "model_lifecycle_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column(
            "action",
            sa.Enum(
                "REGISTERED",
                "PROMOTED",
                "DEMOTED_TO_PREVIOUS",
                "ARCHIVED",
                "ROLLED_BACK",
                name="lifecycle_action",
            ),
            nullable=False,
        ),
        sa.Column("previous_stage", model_stage, nullable=True),
        sa.Column("new_stage", model_stage, nullable=False),
        sa.Column("triggered_by", sa.String(length=50), nullable=False),
        sa.Column("event_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.add_column("model_versions", sa.Column("git_commit", sa.String(length=64), nullable=False))
    op.create_index(
        "uq_model_versions_single_previous",
        "model_versions",
        ["stage"],
        unique=True,
        postgresql_where=sa.text("stage = 'PREVIOUS'"),
    )
    op.create_index(
        "uq_model_versions_single_production",
        "model_versions",
        ["stage"],
        unique=True,
        postgresql_where=sa.text("stage = 'PRODUCTION'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_model_versions_single_production",
        table_name="model_versions",
        postgresql_where=sa.text("stage = 'PRODUCTION'"),
    )
    op.drop_index(
        "uq_model_versions_single_previous",
        table_name="model_versions",
        postgresql_where=sa.text("stage = 'PREVIOUS'"),
    )
    op.drop_column("model_versions", "git_commit")
    op.drop_table("model_lifecycle_events")
    op.execute("DROP TYPE IF EXISTS lifecycle_action")
