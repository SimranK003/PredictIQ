# syntax=docker/dockerfile:1

# PredictIQ backend image. Shared by the "backend" (FastAPI/uvicorn),
# "worker" (Celery), and "migrate" (Alembic, one-shot) Compose services —
# same code, same dependencies, different container `command:`.

FROM python:3.12-slim AS builder

# Build-time only deps for compiling wheels (psycopg2-binary ships
# manylinux wheels so this is mostly a safety net for platforms without
# a matching wheel).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp1 is XGBoost's runtime OpenMP dependency on Linux (the
# equivalent of `brew install libomp` in the native macOS setup —
# see README's Local setup section).
# curl backs the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=appuser:appuser alembic.ini .
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser db ./db
COPY --chown=appuser:appuser ml ./ml
COPY --chown=appuser:appuser workers ./workers
COPY --chown=appuser:appuser scripts ./scripts

# Dataset storage and MLflow's local artifact scratch space are volume
# mount points (see docker-compose.yml) — created here so they exist
# with the right ownership even before a volume is attached.
RUN mkdir -p /app/storage/datasets && chown -R appuser:appuser /app/storage

USER appuser

EXPOSE 8123

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8123/health || exit 1

# WEB_CONCURRENCY controls uvicorn's worker-process count (unset/1 by
# default — identical to every prior phase's single-process behavior;
# Docker Compose doesn't set it). A cloud deployment can raise it for
# real request concurrency — app/services/model_loader.py's production
# model cache was already designed for this ("self-healing by id
# comparison... protects against staleness in a multi-worker
# deployment"), so this isn't new behavior the app doesn't expect, just
# the first place it's actually turned on. This is Docker's *shell*
# CMD form (not the ["executable", "arg"] array/exec form) so the
# ${VAR:-default} substitution below is expanded by /bin/sh — the
# leading `exec` shell builtin then replaces that shell process (PID 1)
# with uvicorn itself, so SIGTERM reaches uvicorn directly for a clean
# shutdown instead of the shell eating it until Docker's kill timeout.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port 8123 --workers ${WEB_CONCURRENCY:-1}
