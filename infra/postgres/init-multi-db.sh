#!/bin/sh
# Runs once, only on a brand-new Postgres data volume (the official
# postgres image executes everything in /docker-entrypoint-initdb.d on
# first init, and never again — see docker-compose.yml's postgres
# service). POSTGRES_DB (from .env) creates the app's own database
# automatically; this script additionally creates a separate `mlflow`
# database on the same server/instance for the mlflow service's
# backend store, so MLflow's run/experiment metadata lives in its own
# database rather than mixed into the app's tables or a SQLite file
# that would need its own volume and has weaker concurrent-writer
# guarantees than Postgres.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE ${MLFLOW_DB_NAME}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${MLFLOW_DB_NAME}')\gexec
EOSQL
