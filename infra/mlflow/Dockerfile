# syntax=docker/dockerfile:1

# Standalone MLflow tracking server. A dedicated small image rather than
# reusing the backend image — the tracking server only needs mlflow
# itself plus a Postgres driver for its backend store, not the whole
# FastAPI/Celery application and its ML dependencies. Version pinned to
# match the mlflow client version backend/requirements.txt uses, since a
# server/client mismatch is exactly the kind of thing that fails subtly.
#
# boto3 is only exercised when --default-artifact-root is an s3:// URI
# (Phase 10 production deployment — see README's Production deployment
# section); Docker Compose/local dev keeps the local-filesystem artifact
# root (/mlartifacts, below) and never touches it.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin mlflow

RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.43.98

RUN mkdir -p /mlartifacts && chown mlflow:mlflow /mlartifacts

USER mlflow
WORKDIR /home/mlflow

EXPOSE 5001

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=10 \
    CMD curl -f http://localhost:5001/health || exit 1

# MLFLOW_BACKEND_STORE_URI is required (no default — an MLflow server
# with no real backing store is a config error, not something to fall
# back silently from). MLFLOW_ARTIFACT_ROOT defaults to this image's
# own local directory (correct for Docker Compose, where a named volume
# is mounted there — see docker-compose.yml) but can be set to an
# s3://bucket/prefix URI for a cloud deployment (see
# render.yaml/README) — the command lives here, in the Dockerfile,
# rather than being duplicated as a `command:`/`dockerCommand:`
# override in every deploy target's own config, so Compose and Render
# can't drift out of sync with each other. Shell form so
# ${VAR:-default} and `exec` (clean SIGTERM handling, same reasoning as
# backend/Dockerfile) both work.
CMD exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:-/mlartifacts}" \
    --host 0.0.0.0 \
    --port 5001
