"""Regression test for the exact bug class Phase 4 hit twice: a process
that talks to MLflow without configuring its tracking URI, silently
falling back to a local ./mlruns file store instead of the real server.

configure_mlflow()/assert_mlflow_configured() are called at startup by
every such process (app/main.py, workers/celery_app.py); this test
proves the detection actually works, independent of any particular
process remembering to call it correctly.
"""

import mlflow
import pytest

from app.core.mlflow_config import assert_mlflow_configured, configure_mlflow


@pytest.fixture(autouse=True)
def _restore_tracking_uri():
    original = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(original)


def test_assert_raises_when_tracking_uri_was_never_configured():
    mlflow.set_tracking_uri("file:///tmp/some-other-mlruns-dir")

    with pytest.raises(RuntimeError, match="MLflow tracking URI mismatch"):
        assert_mlflow_configured()


def test_assert_passes_after_configure_mlflow_is_called():
    mlflow.set_tracking_uri("file:///tmp/some-other-mlruns-dir")

    configure_mlflow()

    assert_mlflow_configured()  # must not raise


def test_configure_mlflow_sets_the_exact_configured_uri():
    from app.core.config import get_settings

    mlflow.set_tracking_uri("file:///tmp/definitely-wrong")
    configure_mlflow()

    assert mlflow.get_tracking_uri() == get_settings().mlflow_tracking_uri
