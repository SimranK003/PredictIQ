"""Tests for the database-level invariants backing the promotion workflow.

These deliberately bypass application logic — the point is to prove the
database itself refuses to hold two "production" (or "previous") rows,
independent of whether app/services/registry.py has a bug. This is what
"do not rely only on application logic" means in practice.

The concurrency test uses two independent, really-committing sessions
(not the shared rollback-per-test db_session fixture, which wraps
everything in one connection's transaction and never truly commits to
other connections — see tests/conftest.py). It cleans up explicitly.
"""

import threading
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import PromotionConflictError
from app.services.registry import promote_model
from db.models import Dataset, ModelStage, ModelVersionRecord
from db.session import SessionLocal


def _always_true(_run_id: str) -> bool:
    return True


def test_database_rejects_a_second_production_row_via_raw_sql(db_session, make_model_version):
    """Bypass promote_model entirely: try to UPDATE a second row straight
    to 'PRODUCTION' with raw SQL and confirm Postgres itself rejects it.
    """
    make_model_version(stage=ModelStage.PRODUCTION)
    second = make_model_version(stage=ModelStage.CANDIDATE)

    with pytest.raises(IntegrityError, match="uq_model_versions_single_production"):
        db_session.execute(
            text("UPDATE model_versions SET stage = 'PRODUCTION' WHERE id = :id"),
            {"id": second.id},
        )
        db_session.flush()


def test_database_rejects_a_second_previous_row_via_raw_sql(db_session, make_model_version):
    make_model_version(stage=ModelStage.PREVIOUS)
    second = make_model_version(stage=ModelStage.CANDIDATE)

    with pytest.raises(IntegrityError, match="uq_model_versions_single_previous"):
        db_session.execute(
            text("UPDATE model_versions SET stage = 'PREVIOUS' WHERE id = :id"),
            {"id": second.id},
        )
        db_session.flush()


@pytest.fixture()
def real_committed_candidates():
    """Two real candidates, committed for real (outside the rollback
    fixture) so two separate connections can race to promote them.
    Cleaned up explicitly afterward.
    """
    session = SessionLocal()
    dataset = Dataset(
        filename="concurrency_test.csv",
        schema_name="telco_customer_churn",
        storage_path="/tmp/concurrency_test.csv",
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        n_rows=100,
        n_columns=21,
        is_valid=True,
        quality_report={"issues": []},
    )
    session.add(dataset)
    session.flush()

    candidates = []
    for i in range(2):
        mv = ModelVersionRecord(
            version_label=f"concurrency-test-v{i}",
            algorithm="logistic_regression",
            mlflow_run_id=f"concurrency-run-{uuid.uuid4().hex}",
            mlflow_experiment_id="1",
            artifact_uri="runs:/fake/model",
            git_commit="deadbeef",
            stage=ModelStage.CANDIDATE,
            metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5},
            params={},
            dataset_id=dataset.id,
        )
        session.add(mv)
        candidates.append(mv)
    session.commit()
    for c in candidates:
        session.refresh(c)
    candidate_ids = [c.id for c in candidates]
    dataset_id = dataset.id
    session.close()

    yield candidate_ids

    cleanup = SessionLocal()
    cleanup.execute(
        text("DELETE FROM model_lifecycle_events WHERE model_version_id = ANY(:ids)"),
        {"ids": candidate_ids},
    )
    cleanup.execute(
        text("DELETE FROM model_versions WHERE id = ANY(:ids)"), {"ids": candidate_ids}
    )
    cleanup.execute(text("DELETE FROM datasets WHERE id = :id"), {"id": dataset_id})
    cleanup.commit()
    cleanup.close()


def test_concurrent_promotions_of_different_candidates_only_one_wins(real_committed_candidates):
    """Two candidates, no existing production. Promote both at the same
    moment from two independent sessions: exactly one should succeed,
    the other must fail with PromotionConflictError (not silently
    produce two production rows) — proving the DB constraint, not just
    the application's row-locking, is what prevents the corrupt state.

    A plain "start two threads" race is too easy to win by accident: on a
    fast local Postgres, thread A can fully commit before thread B even
    starts, making them run *sequentially* (both "succeed" legitimately —
    A becomes production, B's later promote demotes A to previous). The
    barrier forces both threads to pass the promotion-policy check at
    the same synchronized instant, so both read "no current production"
    before either has committed — that's the actual race condition being
    tested.
    """
    candidate_a_id, candidate_b_id = real_committed_candidates
    outcomes = {}
    barrier = threading.Barrier(2)

    def _synced_artifact_checker(_run_id: str) -> bool:
        barrier.wait(timeout=5)
        return True

    def _promote(name: str, candidate_id) -> None:
        session = SessionLocal()
        try:
            promote_model(session, candidate_id, artifact_checker=_synced_artifact_checker)
            outcomes[name] = "success"
        except PromotionConflictError:
            outcomes[name] = "conflict"
        except IntegrityError:
            outcomes[name] = "integrity_error"
        finally:
            session.close()

    thread_a = threading.Thread(target=_promote, args=("a", candidate_a_id))
    thread_b = threading.Thread(target=_promote, args=("b", candidate_b_id))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    results = list(outcomes.values())
    assert results.count("success") == 1, f"expected exactly one success, got {outcomes}"
    assert results.count("conflict") == 1 or results.count("integrity_error") == 1, outcomes

    verify_session = SessionLocal()
    try:
        production_rows = (
            verify_session.query(ModelVersionRecord)
            .filter(ModelVersionRecord.stage == ModelStage.PRODUCTION)
            .filter(ModelVersionRecord.id.in_([candidate_a_id, candidate_b_id]))
            .all()
        )
        assert len(production_rows) == 1, "database ended up with more than one production row"
    finally:
        verify_session.close()
