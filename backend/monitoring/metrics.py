"""Prometheus metric definitions.

All metrics here are incremented/observed only from real request handling
— nothing is pre-populated or hardcoded. An idle service reports zero
counts and an empty histogram, honestly, until real traffic arrives.
"""

from prometheus_client import Counter, Histogram

# Generic HTTP observability, recorded by the request middleware for
# every endpoint (app/core/middleware.py).
HTTP_REQUESTS_TOTAL = Counter(
    "predictiq_http_requests_total",
    "Total HTTP requests handled",
    labelnames=("method", "endpoint", "status_code"),
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "predictiq_http_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "endpoint"),
)

# Prediction-specific business metrics, recorded explicitly by the
# prediction service (app/services/prediction.py) — these answer
# "how many predictions, on which model version, and how many failed,"
# which the generic HTTP counters above can't express.
PREDICTIONS_TOTAL = Counter(
    "predictiq_predictions_total",
    "Total individual predictions produced (single + each batch item)",
    labelnames=("model_version", "algorithm"),
)
# KNOWN LIMITATION: prometheus_client's default registry is per-process
# in-memory state. Single predictions and small (sync) batches run
# inside the API process, so they're correctly reflected here — but
# large batches processed by a separate Celery worker process increment
# this same Counter object in the *worker's own memory*, which the API
# process's GET /metrics can never see. Verified directly: after a real
# 75-record async batch, predictiq_predictions_total only advanced by 6
# (the single + 5-record sync batch that ran in the API process), not 81.
# Properly fixing this needs either prometheus_client's multiprocess
# file-based registry (usually used across multiple workers of the SAME
# service, e.g. gunicorn) or a Pushgateway for the worker to report
# through — both are real infrastructure decisions, not a code fix, and
# out of scope here. A correct production setup would give the Celery
# worker its own /metrics endpoint and scrape it separately.

PREDICTION_ERRORS_TOTAL = Counter(
    "predictiq_prediction_errors_total",
    "Total prediction failures",
    labelnames=("reason",),
)

BATCH_PREDICTIONS_TOTAL = Counter(
    "predictiq_batch_predictions_total",
    "Total POST /predict/batch requests (not individual items)",
    labelnames=("mode",),  # "sync" or "async"
)
