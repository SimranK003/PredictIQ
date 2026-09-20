# PredictIQ — End-to-End ML Prediction & Monitoring Platform

A production-style ML platform demonstrating the full lifecycle of a tabular
classification model — dataset ingestion, reproducible training, experiment
tracking, a real model registry with an explicit promotion workflow,
production inference, prediction logging, and real statistical drift
detection — built around a public dataset (IBM's Telco Customer Churn) with
a FastAPI backend and a Next.js operations dashboard.

This is a personal portfolio project. No proprietary or confidential data is
used anywhere in it.

> **Screenshot note**: this README describes every dashboard page using the
> platform's actual, real data (the same JSON/text captured while manually
> verifying each page against the live backend) rather than embedded image
> files — the tooling used to build this project couldn't mechanically save
> browser screenshots as committed PNGs. Run the app locally (see
> [Local setup (native)](#local-setup-native) or [Docker setup](#docker-setup)) to see it live.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Problem statement](#problem-statement)
3. [Architecture](#architecture)
4. [Tech stack](#tech-stack)
5. [ML workflow](#ml-workflow)
6. [API reference](#api-reference)
7. [Database design](#database-design)
8. [Authentication & authorization](#authentication--authorization)
9. [MLOps workflow: the model registry](#mlops-workflow-the-model-registry)
10. [Monitoring](#monitoring)
11. [Dashboard](#dashboard)
12. [Local setup (native)](#local-setup-native)
13. [Docker setup](#docker-setup)
14. [Production deployment](#production-deployment)
15. [Testing](#testing)
16. [Deployment status](#deployment-status)
17. [Limitations](#limitations)
18. [Future improvements](#future-improvements)
19. [Interview talking points](#interview-talking-points)

---

## Project overview

PredictIQ predicts customer churn for a telecom operator and exposes the
*entire* engineering system around that prediction — not just a model
behind an endpoint. The explicit goal was to build something that reads as
an ML **platform**, where:

- a dataset can be uploaded and is validated before anything touches it,
- training runs asynchronously and logs everything needed to reproduce a
  model later (exact data, exact code version, exact config),
- a model never reaches production by accident — promotion is a deliberate,
  auditable, transactional action,
- every prediction is traceable to the exact model version that produced it,
  even after that version is retired,
- monitoring reports real drift statistics computed from real logged
  predictions, and honestly says "not enough data yet" rather than
  fabricating a number when it doesn't have enough,
- a dashboard makes all of the above legible in under a minute.

## Problem statement

Churn prediction is a standard, well-understood tabular ML problem — which
is deliberate. The interesting engineering here isn't the model; it's
everything *around* the model that a real ML team actually has to build:
lineage, reproducibility, a registry with a real state machine, safe
promotion/rollback, async training that doesn't block an API request, and
monitoring grounded in the platform's own real data.

## Architecture

```mermaid
graph TD
    subgraph Client
        UI[Next.js Dashboard]
    end

    subgraph API["FastAPI (backend/app)"]
        REST[REST API]
        REG[Model Registry Service]
        PRED[Inference Service]
        MON[Monitoring Service]
    end

    subgraph Async["Celery + Redis"]
        WORKER[Celery Worker]
    end

    subgraph ML["ML Layer (backend/ml)"]
        SCHEMA[Shared DatasetSchema]
        PREP[Preprocessing Pipeline]
        TRAIN[Training]
        MLFLOW[(MLflow Tracking)]
    end

    subgraph Data
        PG[(PostgreSQL)]
    end

    UI -->|typed fetch client| REST
    REST --> REG
    REST --> PRED
    REST --> MON
    REST -->|POST /train, /predict/batch| WORKER
    WORKER --> TRAIN
    TRAIN --> SCHEMA
    TRAIN --> PREP
    TRAIN --> MLFLOW
    TRAIN --> PG
    REG --> PG
    PRED -->|loads production pipeline| MLFLOW
    PRED --> PG
    MON --> PG
    MON -->|reference distribution| MLFLOW
```

**Next.js → FastAPI → ML/Database/Monitoring**, concretely:

```mermaid
graph LR
    A[Next.js Dashboard<br/>TypeScript + React Query] -->|HTTPS JSON, NEXT_PUBLIC_API_BASE_URL| B[FastAPI<br/>REST API]
    B --> C[ML Layer<br/>sklearn / XGBoost / MLflow]
    B --> D[(PostgreSQL<br/>lineage + registry + predictions)]
    B --> E[Monitoring<br/>Prometheus + drift stats]
```

The frontend never talks to Postgres, MLflow, or Redis directly — it only
knows the FastAPI base URL (`NEXT_PUBLIC_API_BASE_URL`, a public,
non-secret config value). All real credentials (`DATABASE_URL`,
`MLFLOW_TRACKING_URI`, `REDIS_URL`, etc.) live only in the backend's `.env`,
never in frontend code or environment variables.

### The full lifecycle the dashboard is built to explain

```mermaid
graph TD
    D1[CSV Dataset] -->|POST /datasets, validated| D2[Dataset ID]
    D2 -->|POST /train| D3[Training Job]
    D3 -->|Celery worker| D4[MLflow Experiment]
    D4 --> D5[Model Version — Candidate]
    D5 -->|POST /models/promote, explicit| D6[Production]
    D6 -->|POST /predict| D7[Predictions]
    D7 --> D8[Monitoring: drift, volume, model-version usage]
    D6 -->|POST /models/rollback| D9[Previous]
```

## Tech stack

**Backend**: Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic,
PostgreSQL, Celery, Redis, MLflow, scikit-learn, XGBoost, pandas,
prometheus-client.

**Frontend**: Next.js 16 (App Router), TypeScript, Tailwind CSS 4,
TanStack Query (client-side data fetching/polling), Recharts, Vitest +
React Testing Library.

## ML workflow

```
CSV upload → schema/quality validation → preprocessing (shared, fit-once
Pipeline) → feature engineering → train/val/test split (stratified, no
leakage) → train Logistic Regression / Random Forest / XGBoost → evaluate
(precision, recall, F1, ROC-AUC, confusion matrix — never accuracy alone)
→ log to MLflow (params, metrics, artifact, dataset lineage, git commit)
→ register as a candidate in Postgres
```

The exact same fitted `sklearn.Pipeline` object (feature engineering +
`ColumnTransformer` + classifier) that gets logged during training is what
inference loads and calls `.predict_proba()` on directly — there is no
second, hand-maintained preprocessing implementation at serving time to
drift out of sync with training.

Class imbalance (~26.5% churn in the Telco dataset) is handled via
`class_weight="balanced"` (Logistic Regression, Random Forest) and
`scale_pos_weight` (XGBoost) — not resampling, which would need to be
carefully scoped to the training fold only to avoid leakage.

## API reference

All endpoints are implemented in `backend/app/routers/`. Full interactive
docs are available at `/docs` (FastAPI's generated Swagger UI) whenever the
backend is running.

| Area | Endpoints | Auth |
|---|---|---|
| Auth | `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` | Public (login), session required (logout/me) |
| Datasets | `POST /datasets`, `GET /datasets`, `GET /datasets/{id}` | Session required |
| Training | `POST /train`, `GET /jobs`, `GET /jobs/{id}` | Session required |
| Model registry | `GET /models`, `GET /models/{id}`, `GET /models/{id}/metrics`, `GET /models/production` | Session required |
| Model registry (admin) | `POST /models/promote`, `POST /models/rollback` | Session + **administrator** |
| Inference | `POST /predict`, `POST /predict/batch`, `GET /predictions`, `GET /predictions/{id}` | Session required |
| Monitoring | `GET /monitoring/summary`, `GET /monitoring/drift`, `GET /monitoring/model-usage`, `GET /monitoring/jobs` | Session required |
| Monitoring (scrape) | `GET /metrics` (Prometheus format) | Public — standard scrape convention, see [Authentication & authorization](#authentication--authorization) |
| Health | `GET /health` | Public |

See [Authentication & authorization](#authentication--authorization) for how "session required" is enforced and why.

## Database design

Postgres is the single source of truth for lineage and lifecycle state
(no separate monitoring database — Phase 6 deliberately reused the
existing tables rather than adding one). Core tables:

- **`datasets`** — uploaded CSVs, their validation report, row/column
  counts, and a SHA-256 content hash.
- **`jobs`** — async training and batch-prediction jobs; `queued` →
  `running` → `completed`/`failed`, with `dataset_id`, timestamps, and a
  safe (non-traceback) error message.
- **`model_versions`** — the registry. Each row references its
  `dataset_id`, nullable `training_job_id`, MLflow run ID, git commit,
  snapshotted test metrics/params, and lifecycle `stage`.
- **`model_lifecycle_events`** — an audit trail of every promotion,
  demotion, archive, and rollback.
- **`predictions`** — every real inference result, with the exact
  `model_version_id` that produced it (never just "production").
- **`users`** (Phase 9) — login identity: email, an Argon2id
  `hashed_password` (never a plaintext password), `is_admin`,
  `is_active`. Session state itself is *not* in Postgres — see
  [Authentication & authorization](#authentication--authorization).

Two Postgres **partial unique indexes** enforce "at most one production
model" and "at most one previous model" at the database level — not just
in application code — so a race between two concurrent promotions is
caught by the database itself.

## Authentication & authorization

Added in Phase 9. Every endpoint except `POST /auth/login`, `GET
/health`, and `GET /metrics` requires a valid session; promoting or
rolling back a model additionally requires an administrator session —
see the [API reference](#api-reference) table above for the exact
per-endpoint breakdown.

### Architecture

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as Next.js (client-side)
    participant A as FastAPI
    participant R as Redis
    participant P as Postgres

    B->>F: submit login form
    F->>A: POST /auth/login (email, password)
    A->>P: look up user, verify Argon2id hash
    A->>R: SETEX session:<token> user payload (TTL)
    A-->>F: Set-Cookie (HttpOnly, signed), user JSON
    F-->>B: redirect to /dashboard

    B->>F: navigate to a protected page
    F->>A: GET /models (cookie sent automatically)
    A->>A: verify cookie signature (HMAC, SECRET_KEY)
    A->>R: GET session:<token>
    A->>P: re-fetch user by id (fresh is_admin/is_active)
    A-->>F: 200 (or 401 if no/expired/invalid session)
```

- **Password hashing**: Argon2id via `argon2-cffi`
  (`app/core/security.py`'s `hash_password`/`verify_password`) — the
  current OWASP-recommended algorithm; salted automatically, never
  logged, never returned by any API response.
- **Sessions, not JWTs**: a session is an opaque random token
  (`secrets.token_urlsafe(32)`) stored server-side in Redis
  (`app/services/auth.py`'s `SessionStore`) with a TTL (default 8h,
  `SESSION_TTL_SECONDS`) — chosen over a self-contained JWT specifically
  so logout actually revokes the session immediately (delete the Redis
  key) rather than only "expiring" a token that's still technically
  valid until it times out. This reuses the Redis instance the stack
  already runs for Celery — no new infrastructure.
- **Cookie**: `HttpOnly` (unreadable by JS, so an XSS bug can't steal it
  via `document.cookie`), `SameSite=Lax` (default; configurable via
  `SESSION_COOKIE_SAMESITE`), and `Secure` wherever the app is actually
  served over HTTPS (`SESSION_COOKIE_SECURE`, `false` by default for
  local HTTP dev — see [Limitations](#limitations)).
  The cookie *value* is also HMAC-signed with `SECRET_KEY`
  (`app/core/security.py`'s `sign_token`/`unsign_token`) so a
  tampered/forged cookie is rejected before ever touching Redis or the
  database — the real security boundary is still "does this token exist
  as a live session in Redis," not the signature, but the signature
  makes garbage/forged input cheap to reject.
- **Re-verified every request**: `get_current_user`
  (`app/core/security.py`) re-reads the user row from Postgres on every
  request, not just the Redis-cached session payload — deactivating a
  user or revoking admin takes effect on their very next request, not
  only after their session expires.
- **Authorization model**: exactly two levels — authenticated user and
  administrator (`users.is_admin`). No roles table, no per-resource
  permissions: the brief was "don't build RBAC the requirements don't
  need," and a boolean is all two levels need. `require_admin`
  (`app/core/security.py`) gates `POST /models/promote` and `POST
  /models/rollback` specifically, since those are the only two actions
  that change what's actually serving production traffic; everything
  else (viewing models/predictions/monitoring, submitting a dataset,
  submitting a training job, running a prediction) only requires being
  logged in — training only ever produces a *candidate*
  (`app/services/registry.py`), never touches production on its own, so
  it doesn't need the admin gate promotion does.
- **Backend-enforced, not frontend-enforced**: every one of the checks
  above runs in FastAPI dependencies
  (`get_current_user`/`require_admin`), independent of anything the
  Next.js app does. The frontend's own route guard
  (`frontend/src/components/auth/AppShell.tsx`) and the Promote/Rollback
  buttons hiding for non-admins
  (`frontend/src/components/models/PromotionControls.tsx`) are UX
  conveniences layered on top — verified directly with `curl` against
  the API (no browser, no frontend) during this phase, not just inferred
  from the UI behaving correctly.
- **Login rate limiting**: a Redis-backed fixed-window counter
  (`app/services/auth.py`), keyed by `email:client_ip`, blocking further
  attempts with `429` after `LOGIN_RATE_LIMIT_MAX_ATTEMPTS` (default 5)
  failures within `LOGIN_RATE_LIMIT_WINDOW_SECONDS` (default 300) — a
  successful login clears the counter. No new infrastructure (reuses the
  existing Redis instance); this is a basic brute-force speed bump, not
  a substitute for a real account-lockout/anomaly-detection system a
  production deployment would want.
- **No user enumeration**: a wrong password and a nonexistent email both
  return the identical `401 "Incorrect email or password."`, and both
  pay the same real Argon2 hashing cost (`app/routers/auth.py`'s
  `_DUMMY_HASH`) so a timing difference can't distinguish them either.
- **No password leakage in error responses**: FastAPI's default 422
  validation-error handler echoes back each invalid field's raw
  submitted value — which would reflect an attempted password into the
  HTTP response for a malformed `password` field. `app/main.py`'s
  `RequestValidationError` handler redacts any field named `password`
  before the response is ever serialized (see
  `tests/integration/test_auth_api.py::test_password_is_never_echoed_back_in_a_validation_error`).

### Local development credentials & bootstrapping the first admin

There is deliberately no self-registration endpoint — accounts are
provisioned with `backend/scripts/create_admin.py`, which is also how
you create the first administrator after migrations:

```bash
cd backend
python -m scripts.create_admin --email you@example.com --admin
```

It prompts for the password via `getpass` (not echoed to the terminal,
never accepted as a bare CLI argument, which would otherwise land in
shell history). Re-running it for an existing email updates that user's
password/admin flag instead of erroring — handy for resetting a
forgotten local password. Omit `--admin` to create a regular user.

In Docker:

```bash
docker compose run --rm backend python -m scripts.create_admin --email you@example.com --admin
```

### Environment variables

All defined in `app/core/config.py` and documented (with the same
defaults) in `backend/.env.example` (native) and the root `.env.example`
(Docker):

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | `dev-insecure-secret-key-change-me` | HMAC-signs the session cookie. The default is an obvious placeholder — generate a real one (`python -c "import secrets; print(secrets.token_hex(32))"`) for anything beyond a throwaway local instance. |
| `SESSION_COOKIE_NAME` | `predictiq_session` | |
| `SESSION_TTL_SECONDS` | `28800` (8h) | Session lifetime, enforced by Redis's own key expiry. |
| `SESSION_COOKIE_SECURE` | `false` | Must be `true` wherever served over HTTPS — `false` is required for local plain-HTTP dev, since a `Secure` cookie is silently dropped by the browser over HTTP. |
| `SESSION_COOKIE_SAMESITE` | `lax` | |
| `LOGIN_RATE_LIMIT_MAX_ATTEMPTS` | `5` | Failed attempts allowed per window before `429`. |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | `300` | |

## MLOps workflow: the model registry

MLflow remains the source of truth for experiment tracking (params,
metrics, the model artifact). `model_versions` in Postgres is the
*application-level* lifecycle on top of it — `candidate` → `production` →
`previous` → `archived` — driven only by explicit API calls, never
automatically:

- Training always produces **candidates**. A candidate never becomes
  production on its own.
- `POST /models/promote` is gated by a configurable policy: the candidate
  must have a valid linked dataset, all required metrics present, and
  (optionally) a live-verified MLflow artifact.
- Promoting demotes the current production model to `previous` and
  archives whatever was previously `previous` — all in one transaction,
  with the database's partial unique indexes as the real backstop against
  a race producing two production models.
- `POST /models/rollback` swaps production and previous back.

## Monitoring

See [`docs/monitoring.md`](docs/monitoring.md) for the full methodology
write-up (why KS for numeric features, why PSI for categorical, threshold
definitions, minimum-sample-size reasoning, and the reference-distribution
lineage). In short:

- **Reference distribution**: computed on demand from the exact training
  dataset of the model version being checked (`model_versions.dataset_id`
  → the real CSV on disk), never a generic/unrelated dataset.
- **Numeric drift**: two-sample Kolmogorov–Smirnov test.
- **Categorical drift**: Population Stability Index over category
  proportions, with epsilon smoothing for categories unseen in one side.
- **Minimum sample size**: configurable (`DRIFT_MIN_SAMPLES`, default 100)
  — below it, the API returns `"status": "insufficient_data"` with the
  actual count, never a fabricated score.
- **Model-version isolation**: drift for one model version never mixes in
  predictions from a different (e.g. superseded) version.

## Dashboard

Seven pages, all built against the real backend — nothing here is
hardcoded demo data:

- **`/dashboard`** — production model, system health, prediction volume,
  drift status, active/failed training jobs, all from
  `GET /monitoring/summary` + `GET /health`. Explicitly labels
  process-scoped Prometheus counters as such (e.g.
  `total_predictions_this_process`) rather than presenting them as a
  global total — the real backend value from `GET /monitoring/summary`
  distinguishes real database-backed prediction counts from in-process
  Prometheus counters, and the UI preserves that distinction.
- **`/models`** — every model version with real metrics (ROC-AUC,
  precision, recall, F1), lifecycle stage badges, dataset, and training
  job, filterable by stage.
- **`/models/[id]`** — full metadata (MLflow run, git commit, dataset
  content hash, training config), a five-step lineage diagram (Dataset →
  Training Job → MLflow Run → Model Version → Predictions), the confusion
  matrix, and explicit Promote/Rollback actions — each behind a
  confirmation dialog that shows exactly what will change before
  anything happens.
- **`/jobs`** — every training/batch-prediction job, polling active ones
  every 4s, with real durations and the model versions each job produced.
- **`/predictions`** — real inference history, paginated, filterable by
  model version and date. Raw `input_features` are shown only in the
  per-prediction detail drawer (opened by clicking a row), never in the
  list table.
- **`/monitoring`** — prediction volume and model-version distribution,
  plus the full drift breakdown (feature, test, score, threshold, status)
  with a chart and an explicit note that drift measures input
  distribution change, not model accuracy.
- **`/train`** — submit a real async training job (dataset, algorithms,
  hyperparameters) and watch it progress queued → running →
  completed/failed in place, no page reload.

Every page has real loading, error, and empty states (a `role="status"`
spinner, a `role="alert"` error with retry, and a labeled empty state) —
never a blank screen.

## Local setup (native)

This section runs every process directly on the host (Homebrew Postgres/
Redis, a local MLflow server process, a Celery worker process) — the way
this project was originally developed and tested. See
[Docker setup](#docker-setup) below for the containerized equivalent.

### Prerequisites

- Python 3.11+ (developed against 3.12)
- Node.js 20+
- PostgreSQL and Redis running locally (this project was developed and
  tested against Homebrew-installed `postgresql@16` and `redis` on macOS,
  **not** Docker — see [Deployment status](#deployment-status))
- macOS only: `brew install libomp` (XGBoost needs it; without it, `import
  xgboost` segfaults on load)

### Backend

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + pytest/ruff — see that file
cp .env.example .env   # edit DATABASE_URL/REDIS_URL/MLFLOW_TRACKING_URI as needed
alembic upgrade head
python -m scripts.create_admin --email you@example.com --admin  # first login — see Authentication & authorization
uvicorn app.main:app --reload --port 8123
```

A real MLflow tracking server (not the client's local file-store default)
and a Celery worker are both required for full functionality:

```bash
# in a second terminal, from backend/
mlflow server --backend-store-uri sqlite:///mlruns_server/mlflow.db \
  --default-artifact-root ./mlruns_server/artifacts --host 127.0.0.1 --port 5001

# in a third terminal, from backend/ (--pool=solo is a macOS-only
# workaround — Celery's default prefork pool can segfault on macOS when
# forking a process that already loaded XGBoost/numpy/libomp)
celery -A workers.celery_app worker --loglevel=info --pool=solo
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL, defaults to http://localhost:8123
npm run dev
```

Visit `http://localhost:3000` (redirects to `/dashboard`).

## Docker setup

Phase 8 added a Docker Compose stack that runs the same architecture as
[Local setup (native)](#local-setup-native) — FastAPI, a Celery worker,
Postgres, Redis, MLflow, and the Next.js frontend — each in its own
container, talking to each other by Compose service name instead of
`localhost`. It does not replace the native setup above; both are
documented and both work.

### What's different from native, and why

- **MLflow's backend store is PostgreSQL, not SQLite.** Native dev uses
  `sqlite:///mlruns_server/mlflow.db` (see above). In Docker, MLflow uses
  a second logical database (`mlflow`, created alongside the app's own
  `predictiq` database by
  [`infra/postgres/init-multi-db.sh`](infra/postgres/init-multi-db.sh))
  on the *same* Postgres server — reusing the one stateful data service
  the stack already needs, and giving MLflow's metadata store real
  concurrent-writer guarantees instead of SQLite's. This is an explicit,
  documented choice, not a silent behavior change: MLflow's tracking API
  and the application's use of it (`MLFLOW_TRACKING_URI`,
  `mlflow.set_tracking_uri`/`set_experiment` in
  `app/core/mlflow_config.py`) are identical either way — only the engine
  backing MLflow's own metadata differs.
- **Artifact storage stays filesystem-based**, on a named volume
  (`mlflow_artifacts`, mounted at `/mlartifacts` in the `mlflow`
  container) rather than S3/GCS — matching native dev's
  `./mlruns_server/artifacts`. The backend and worker containers never
  mount this volume themselves: MLflow's server proxies artifact
  read/writes over HTTP by default (`mlflow server`'s "serve-artifacts"
  behavior), so `runs:/<run_id>/model` URIs (see
  `backend/ml/training/train.py`) resolve correctly over the network
  alone, the same way `app/services/model_loader.py`'s
  `mlflow.sklearn.load_model(...)` call already expects.
- **The Celery worker uses Linux's default prefork pool**, not
  `--pool=solo`. `--pool=solo` in the native instructions works around a
  macOS-specific fork-after-loading-XGBoost/libomp segfault (see
  [Limitations](#limitations)) that doesn't occur on Linux, so the
  workaround isn't carried into the container image.
- **Dataset storage is a named volume** (`dataset_storage`, mounted at
  `/app/storage` in both the `backend` and `worker` containers) instead
  of a host-relative `storage/datasets` directory — the worker's async
  training path (`app/services/training.py`) reads the same
  `dataset.storage_path` the API wrote when a CSV was uploaded
  (`app/services/ingestion.py`), so both containers mount the identical
  path.
- **Database migrations run in a dedicated one-shot `migrate` service**
  (`alembic upgrade head`, then exits) that `backend` and `worker` both
  wait on (`condition: service_completed_successfully`) before starting
  — the schema is never raced by two containers migrating concurrently,
  and startup never assumes a schema that isn't actually there yet.
  Alembic's migration files under `backend/db/migrations/` remain the
  only source of truth for schema changes; nothing recreates or
  hand-edits the schema.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2
  (`docker compose version`)

### Environment setup

```bash
cp .env.example .env
```

`.env.example` documents every variable Compose reads — Postgres
credentials, per-service host ports, MLflow's experiment name, CORS
origins, and `NEXT_PUBLIC_API_BASE_URL`. All defaults are non-secret
local-development values. This file is independent of
`backend/.env(.example)` and `frontend/.env.local(.example)`, which only
apply to the native setup — Compose does not read those.

### Starting the stack

```bash
docker compose up --build
```

This builds four images (`backend`, `worker` — same image, different
`command:` — `mlflow`, `frontend`) and starts, in dependency order:
Postgres and Redis → MLflow → the one-shot `migrate` job → the backend
API and Celery worker → the frontend. `depends_on` conditions
(`service_healthy` / `service_completed_successfully`) mean a service
only starts once what it depends on is actually ready, not merely
started — so, for example, the backend never starts against a database
schema that hasn't been migrated yet.

Run it detached with `docker compose up --build -d`, and follow logs for
one service with `docker compose logs -f backend` (or `worker`, `mlflow`,
etc).

### Service URLs (host machine)

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API / docs | http://localhost:8123 (`/docs` for Swagger UI) |
| MLflow UI | http://localhost:5001 |
| Postgres | `localhost:5432` (`psql`/GUI client, e.g. `psql postgresql://predictiq:predictiq@localhost:5432/predictiq`) |
| Redis | `localhost:6379` (`redis-cli -p 6379`) |

Postgres and Redis are published to the host only for local developer
convenience (inspecting data, debugging) — every container-to-container
connection uses the Compose service name (`postgres`, `redis`, `mlflow`,
`backend`), never `localhost`, since `localhost` inside a container means
that container itself.

### Stopping / restarting

```bash
docker compose down          # stop and remove containers; volumes persist
docker compose down -v       # also delete the named volumes (destroys all data)
docker compose up -d         # restart without rebuilding
docker compose up --build -d # rebuild images first (after a code/dependency change)
```

### Persistent volumes

| Volume | Mounted at | Contains | Survives `down`? | Survives `down -v`? |
|---|---|---|---|---|
| `postgres_data` | `postgres:/var/lib/postgresql/data` | App tables + MLflow's backend-store tables | Yes | No |
| `mlflow_artifacts` | `mlflow:/mlartifacts` | Logged model artifacts (`mlflow.sklearn.log_model`) | Yes | No |
| `dataset_storage` | `backend:/app/storage`, `worker:/app/storage` | Uploaded dataset CSVs | Yes | No |

A plain `docker compose restart` or `down`/`up` cycle keeps all three —
only `down -v` (or manually deleting a volume) discards data.

### Migrations

Handled automatically by the `migrate` service on every `docker compose
up`. To run Alembic manually against the containerized database (e.g. to
check current revision, or generate a new migration):

```bash
docker compose run --rm migrate alembic current
docker compose run --rm migrate alembic upgrade head
```

### Creating the first admin user

There's no self-registration page — after the stack is up, create an
account the same way as native dev, just run through the `backend`
image (see [Authentication & authorization](#authentication--authorization)
for the full picture):

```bash
docker compose run --rm backend python -m scripts.create_admin --email you@example.com --admin
```

### Running tests against the containerized stack

The test suites themselves aren't containerized (no dedicated `test`
service) — they still run from the host, the same as
[Testing](#testing) describes, but can point at the Dockerized Postgres/
Redis by using the host-published ports:

```bash
# Backend, from backend/ with the project's virtualenv active
DATABASE_URL=postgresql+psycopg2://predictiq:predictiq@localhost:5432/predictiq \
REDIS_URL=redis://localhost:6379/0 \
pytest

# Frontend — unaffected by which backend is running, since it's all
# component/unit tests against mocked API responses
cd frontend && npm test
```

### Troubleshooting

- **A service is stuck "starting" / dependents never start**: check its
  healthcheck logs with `docker compose ps` (shows health status) and
  `docker compose logs <service>`. `mlflow` needs `postgres` healthy
  first; `backend`/`worker` need `postgres`, `redis`, `mlflow` healthy
  *and* `migrate` to have exited 0.
- **`migrate` fails on first run**: usually means `postgres` wasn't
  actually ready despite its healthcheck (rare) or `DATABASE_URL` doesn't
  match the `POSTGRES_*` values in `.env` — check
  `docker compose logs postgres migrate`.
- **Frontend shows network errors calling the API**: `NEXT_PUBLIC_API_BASE_URL`
  is baked into the frontend image at *build* time (browser-side code
  can't resolve Compose service names). If you changed
  `BACKEND_HOST_PORT` or `NEXT_PUBLIC_API_BASE_URL` in `.env`, rebuild
  the frontend: `docker compose up --build frontend`.
- **Port already in use**: another process (possibly the native setup
  from [Local setup (native)](#local-setup-native)) is already bound to
  5432/6379/5001/8123/3000. Stop it, or change the corresponding
  `*_HOST_PORT` / `NEXT_PUBLIC_API_BASE_URL` in `.env`.
- **`docker compose down -v` then `up` and old data/models are gone**:
  expected — `-v` deletes the named volumes (Postgres data, MLflow
  artifacts, dataset storage). Omit `-v` to keep them.
- **XGBoost import errors inside the backend/worker container**: the
  image installs `libgomp1` (Linux's OpenMP runtime, the equivalent of
  native macOS's `brew install libomp`) — if this is missing it means the
  image wasn't rebuilt after a Dockerfile change; `docker compose build
  --no-cache backend`.

## Production deployment

Phase 10. Everything in this section is **STATICALLY VALIDATED**, not
**VERIFIED IN CLOUD** — see [Verification status](#verification-status)
at the end of this section for the precise, honest breakdown of what
was and wasn't actually exercised, and why (no Docker install and no
cloud credentials were available in the environment this was authored
in — see [Docker setup](#docker-setup)'s own note on the same
constraint). Nothing here claims a live deployment, uptime, or
production traffic that didn't happen.

### Why Render

A single managed PaaS (Render) rather than hand-wiring individual AWS/
GCP services or adding Kubernetes — this is a portfolio-scale app with
7 logical services (matching `docker-compose.yml` exactly: Postgres,
Redis, MLflow, backend, worker, migrate, frontend), and Render's
Blueprint model (`render.yaml`, below) maps onto that 1:1 with managed
Postgres and managed Redis-compatible Key Value, private services with
no public URL, one declarative file, and free HTTPS — without
introducing infrastructure this project doesn't need. No Kubernetes:
there's no multi-region, autoscaling, or service-mesh requirement here
that would justify it.

### Production architecture

```mermaid
graph TD
    subgraph Public["Public (HTTPS)"]
        FE[Next.js frontend<br/>predictiq-frontend]
        BE[FastAPI backend<br/>predictiq-backend]
    end
    subgraph Private["Private network only"]
        WK[Celery worker<br/>predictiq-worker]
        ML[MLflow tracking server<br/>predictiq-mlflow]
        RD[(Redis / Key Value<br/>predictiq-redis)]
        PG[(PostgreSQL<br/>predictiq-postgres)]
    end
    subgraph Persistent
        PG
        MLDISK[(MLflow artifact disk)]
        S3[(S3-compatible bucket<br/>dataset CSVs)]
    end

    Browser -->|HTTPS| FE
    FE -->|HTTPS, browser-side fetch| BE
    BE --> PG
    BE --> RD
    BE -->|HTTP, private network| ML
    BE --> S3
    WK --> PG
    WK --> RD
    WK -->|HTTP, private network| ML
    WK --> S3
    ML --> PG
    ML --> MLDISK
```

This is the same topology `docker-compose.yml` already runs locally —
see [Docker setup](#docker-setup) — just with each service hosted by
Render instead of a container on one host, and Postgres/Redis as
managed offerings instead of containers Render/you would otherwise have
to operate.

### Services deployed

| Service | Render type | Public? | Persistent storage | Purpose |
|---|---|---|---|---|
| `predictiq-frontend` | `web` (Docker) | Yes (HTTPS) | — | Next.js dashboard |
| `predictiq-backend` | `web` (Docker) | Yes (HTTPS) | — (dataset files go to S3, not a disk) | FastAPI API |
| `predictiq-worker` | `worker` (Docker) | No | — | Celery: async training + batch prediction |
| `predictiq-mlflow` | `pserv` (Docker) | No | 5GB disk (`/mlartifacts`) | MLflow tracking server |
| `predictiq-redis` | `keyvalue` | No (`ipAllowList: []`) | Depends on plan | Celery broker/results + session store |
| `predictiq-postgres` | Render Postgres | No (`ipAllowList: []`) | 1GB disk (managed) | App tables + MLflow backend store |

All defined declaratively in [`render.yaml`](render.yaml) at the repo
root — see its own extensive inline comments for the reasoning behind
every field, especially the ones that needed real research against
Render's documented Blueprint schema rather than assumption (disk
sharing, `fromService`/`fromDatabase` reference syntax, `ipAllowList`
semantics, how Docker build args reach `NEXT_PUBLIC_API_BASE_URL`).

### What's different from Docker Compose, and why

- **Dataset storage moves to S3-compatible object storage** (new in
  Phase 10 — see `app/services/dataset_storage.py`). Docker Compose's
  `backend` and `worker` containers share one named volume
  (`dataset_storage`); Render's persistent disks attach to exactly one
  service each (confirmed against Render's own docs while building
  this), so there's no equivalent of that shared volume across two
  separate Render services. `DATASET_STORAGE_BACKEND=s3` (set in
  `render.yaml`) switches both services to read/write the same S3
  bucket instead — `DATASET_STORAGE_BACKEND=local` (the default,
  unchanged for native dev and Docker Compose) still works exactly as
  before. This required a real, if small, code change — see
  [File storage](#file-storage) below.
- **MLflow keeps a local disk**, not S3 — unlike dataset storage, only
  one service (`predictiq-mlflow` itself) ever touches its artifact
  directory; the backend/worker only ever reach it over HTTP (MLflow's
  own "serve-artifacts" proxying, same as in Docker Compose), so
  there's no cross-service sharing problem here and no need for the
  extra moving part. `infra/mlflow/Dockerfile`'s `boto3` install means
  switching to `s3://...` via `MLFLOW_ARTIFACT_ROOT` remains a one-line
  config change later if you outgrow a single disk.
- **Redis broker and result-backend share one logical database**,
  unlike Docker Compose's db 0 (broker) / db 1 (results) split — Render
  Key Value's connection string has no selectable db-index in the form
  `render.yaml` can reference. This is a deliberate, documented
  simplification (sharing one Redis namespace for both is a completely
  standard Celery configuration), not an oversight.
- **`SESSION_COOKIE_SECURE=true`** (native/Compose default `false`) —
  Render terminates TLS for every `web` service, so the session cookie
  (Phase 9) is always sent over HTTPS in production; `false` would
  silently work but defeat the point of the flag.
- **`WEB_CONCURRENCY=2`** runs the backend with 2 uvicorn worker
  processes instead of Docker Compose's 1 — see `backend/Dockerfile`'s
  comment on why `app/services/model_loader.py`'s production-model
  cache was already designed to tolerate this.
- **`TRUST_PROXY_HEADERS=true`** (native/Compose default `false`) — see
  [Security review](#security-review)'s "Rate limiter behind a proxy"
  row for the real bug this fixes.
- **The `migrate` one-shot Compose service has no Render equivalent** —
  `predictiq-backend`'s `preDeployCommand: alembic upgrade head` runs
  before every deploy instead (Render-native mechanism for exactly this
  purpose), so there's still exactly one migration path, just expressed
  differently per platform.

### File storage

| Data | Native dev | Docker Compose | Render (production) |
|---|---|---|---|
| Uploaded dataset CSVs | `backend/storage/datasets/` on the host disk | `dataset_storage` named volume (shared by backend+worker) | S3-compatible bucket (`DATASET_STORAGE_BACKEND=s3`) |
| MLflow artifacts | `backend/mlruns_server/artifacts/` on the host disk | `mlflow_artifacts` named volume | `predictiq-mlflow`'s own Render disk |
| MLflow/app metadata | SQLite file / local Postgres | Postgres (2 logical DBs — see Docker setup) | Render managed Postgres (2 logical DBs) |

`app/services/dataset_storage.py` is the single place that branches on
`DATASET_STORAGE_BACKEND` — `save_dataset_bytes`, `dataset_file_exists`,
`read_dataset_bytes`, `read_dataset_dataframe` are used by
`app/services/ingestion.py` (write path) and `app/services/training.py`
/ `ml/training/train.py` (read + content-hash-verify path) instead of
raw `Path` calls. The local-filesystem behavior these replaced is
byte-for-byte unchanged when `DATASET_STORAGE_BACKEND=local` (the
default) — this was verified by re-running the full existing test suite
(unchanged) after the refactor, not just by inspection.

### Environment variables & secrets

Every variable is declared in [`render.yaml`](render.yaml); anything
with `sync: false` is a secret Render prompts you to enter by hand
during Blueprint creation — **it is never written to this repo**. At
minimum, be ready to provide:

| Secret | Used by | Notes |
|---|---|---|
| `SECRET_KEY` | backend | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DATASET_STORAGE_S3_BUCKET`, `_REGION`, `_ENDPOINT_URL` | backend, worker | Endpoint URL only needed for a non-AWS S3-compatible provider (e.g. Cloudflare R2) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | backend, worker | Scope this credential to only the one dataset bucket if your provider supports it |

`DATABASE_URL` and Redis's connection string are **not** manually
entered — `render.yaml`'s `fromDatabase`/`fromService` references
inject them automatically from the managed Postgres/Key Value instances
Render itself provisions, so those credentials never pass through your
hands or this repo at all.

### Deploying (exact steps)

1. Push this repo to GitHub (a remote is already configured —
   `git remote -v`) if you haven't already.
2. In the [Render dashboard](https://dashboard.render.com), choose
   **New > Blueprint** and point it at the repo. Render parses
   `render.yaml` and prompts for every `sync: false` secret listed
   above.
3. Confirm and let the initial deploy run. `predictiq-postgres` and
   `predictiq-redis` provision first; `predictiq-mlflow`,
   `predictiq-backend` (which runs `alembic upgrade head` via
   `preDeployCommand` before serving), `predictiq-worker`, and
   `predictiq-frontend` follow.
4. Create the first admin user — same command as
   [Docker setup](#docker-setup), run against the live service instead:
   open a Render Shell on `predictiq-backend` and run
   `python -m scripts.create_admin --email you@example.com --admin`.
5. Verify `NEXT_PUBLIC_API_BASE_URL` and `API_CORS_ORIGINS` actually
   match the URLs Render assigned (`render.yaml`'s values are
   *predicted* from each service's `name:` — see its comments; if you
   renamed a service or attached a custom domain, update these and
   redeploy the **frontend** specifically, since
   `NEXT_PUBLIC_API_BASE_URL` is baked into the JS bundle at build time
   and a plain env-var change alone won't retroactively fix an
   already-built bundle).
6. Visit `predictiq-frontend`'s URL, log in, and work through
   [Real deployment verification](#real-deployment-verification-checklist)
   below.

### Database migration procedure

`alembic upgrade head` runs as `predictiq-backend`'s
`preDeployCommand` — before every deploy, not just the first one — so
the schema is never more than one migration behind the code that's
about to start serving. This is the same Alembic-only, additive-only
discipline every earlier phase used (see
[Database design](#database-design)): migrations only ever add/alter,
`alembic upgrade head` is the only path to change the schema, and
nothing in this deployment recreates or hand-edits it.

**Rollback considerations**: Alembic supports `alembic downgrade
<revision>`, but every migration in this project so far is additive
(new tables/columns/indexes — see `backend/db/migrations/versions/`),
and none has a downgrade path that's actually been exercised against
real data. Treat `alembic downgrade` here as a documented capability,
not a tested one: **the safer rollback for a bad deploy is Render's own
"Redeploy previous version" on `predictiq-backend`**, which rolls back
the application code while leaving the (additively-migrated, backward-
compatible) schema alone — reserve a real `alembic downgrade` for a
migration that turns out to be actively harmful, and test it against a
copy of production data first, not production itself.

### CI/CD pipeline

[`​.github/workflows/ci.yml`](.github/workflows/ci.yml) — this repo had
no existing GitHub Actions workflows before Phase 10, so this is new,
not an extension of something pre-existing. Four jobs, every one gating
the next:

1. **`backend`** — real Postgres + Redis service containers (GitHub-
   hosted runner), `ruff check .`, `alembic upgrade head` against the
   ephemeral database, then the full `pytest` suite, then a
   non-blocking `pip-audit` scan (see [Security review](#security-review)).
2. **`frontend`** — `eslint`, `tsc --noEmit`, `vitest run`, `next build`.
3. **`docker-build`** — builds all three production images (backend,
   frontend, mlflow) on the runner, push disabled. This is the one
   thing this environment's own missing Docker install couldn't verify
   directly while authoring these Dockerfiles (Phase 8) — GitHub's
   hosted runners have Docker natively, so this job is real, working
   CI coverage for exactly that gap, even though it hasn't executed
   yet in *this* environment (see [Verification status](#verification-status)).
4. **`deploy`** — only runs on a push to `main`, only after the other
   three succeed, and only does anything if a `RENDER_DEPLOY_HOOK_URL`
   repo secret is set (get this URL from `predictiq-backend`'s Render
   dashboard under Settings > Deploy Hook, and add it under
   `github.com/<you>/PredictIQ/settings/secrets/actions` — a deploy
   hook triggers Render to pull and redeploy the latest `main`, which
   already re-runs `preDeployCommand`, i.e. migrations). Without that
   secret configured, this job exits cleanly without deploying anything
   — deployment is opt-in, not automatic on a fresh clone of this repo.

This workflow has **not** actually run on GitHub Actions as part of
building it (no push was made from this environment) — it's syntax-
validated (parsed as YAML) but not execution-verified. See
[Verification status](#verification-status).

### Observability

Nothing new was built here — Phase 10 wires up what Phase 4/6 already
built, for a cloud audience instead of a local one:

- **Health**: `GET /health` on `predictiq-backend`'s public URL (also
  what Render's own `healthCheckPath` polls to decide if the service is
  up) — see [Health](#api-reference) in the API reference for exactly
  what "ok"/"degraded"/"unhealthy" mean.
- **Logs**: Render's dashboard has a live/historical log stream per
  service (`predictiq-backend`, `predictiq-worker`, `predictiq-mlflow`,
  `predictiq-frontend` each separately) — this project's structured
  JSON logging (`app/core/logging.py`) means each line is already
  machine-parseable there, same as local dev.
- **Model/data drift**: `GET /monitoring/drift` (session required — see
  [Authentication & authorization](#authentication--authorization)) or
  the dashboard's `/monitoring` page, exactly as documented in
  [Monitoring](#monitoring) — nothing cloud-specific changes here.
- **Prometheus**: `GET /metrics` remains intentionally unauthenticated
  (see [API reference](#api-reference)) but **is not scraped by
  anything in this deployment** — Render doesn't run a Prometheus
  server for you, and none was added here (would be unnecessary
  infrastructure for a portfolio deployment — see
  [Limitations](#limitations) for the existing per-process-counter
  caveat this doesn't change). The endpoint exists and returns real
  data if you point your own Prometheus/Grafana at it later.

### Security review

Performed against the actual code/config, not a checklist filled in
from memory:

| Area | Finding |
|---|---|
| CORS | `allow_origins` is an explicit list from `API_CORS_ORIGINS` (never `"*"`), `allow_credentials=True` — required for the session cookie to work cross-origin at all, and only safe *because* origins are explicit, not wildcarded (`app/main.py`, unchanged since Phase 9). |
| Cookies | `HttpOnly`, `SameSite=Lax`, and (new in Render's `render.yaml`) `SESSION_COOKIE_SECURE=true` — see [What's different from Docker Compose](#whats-different-from-docker-compose-and-why). |
| Auth secret | `SECRET_KEY` is `sync: false` in `render.yaml` — never defaults to the checked-in placeholder in production; you're prompted for a real one at Blueprint creation. |
| Exposed ports/services | Postgres and Redis both set `ipAllowList: []` — confirmed via Render's own docs (fetched while writing this file) that this is the documented way to block all public internet access to a managed database/Key Value instance, not just an assumption. `predictiq-mlflow` and `predictiq-worker` are `pserv`/`worker` types, which have no public URL by construction. |
| Debug flags | No `--reload` in any production `CMD`/`dockerCommand` (checked `backend/Dockerfile`, `render.yaml`); `ENVIRONMENT=production` set explicitly. |
| Secrets in the repo | `git check-ignore` confirms `.env`/`.env.local` stay untracked; grepped this diff for `SECRET_KEY=`, `AWS_SECRET`, `PASSWORD=` literals — only the documented placeholder values in `.env.example` files. |
| Docker images | Non-root users in all three custom images (`backend`, `frontend`, `infra/mlflow`), unchanged since Phase 8. |
| Dependency vulnerabilities | `pip-audit` (backend) found real advisories on 4 packages: see below. `npm audit` (frontend) reports **0 vulnerabilities**. |
| Error responses | Unhandled exceptions never leak a traceback (`app/main.py`'s catch-all handler, Phase 1); validation errors redact any `password`-named field (Phase 9). Unchanged, re-verified this phase. |
| DB connectivity | `pool_pre_ping=True` (`db/session.py`) — a dropped connection (e.g. a managed Postgres failover) is detected and replaced on next use rather than surfacing as a raw driver error. |
| Public/private boundaries | Matches the brief's PUBLIC/PRIVATE split exactly — see [Services deployed](#services-deployed). |
| Rate limiter behind a proxy | **Real bug found and fixed this phase**: the login rate limiter (Phase 9) keyed on `request.client.host`, which — once actually deployed behind Render's edge proxy — would key on the proxy's address for every user, collapsing everyone into one shared bucket. `TRUST_PROXY_HEADERS` (`app/core/config.py`, `false` by default, `true` only in `render.yaml`) makes the limiter honor `X-Forwarded-For` only when the deployment topology actually guarantees it's trustworthy. Covered by new tests in `tests/integration/test_auth_api.py`. |

**Dependency vulnerabilities (`pip-audit -r backend/requirements.txt`,
run in this environment, real output)**:

- **`python-multipart` 0.0.9 → upgraded to 0.0.32 this phase** (all
  reported advisories resolved) — verified against the full 213-test
  backend suite before and after, no behavior change. This is the
  package parsing every dataset upload's multipart body, so it was
  worth fixing now rather than deferring.
- **`starlette`** (pinned transitively by `fastapi==0.115.0`'s
  `starlette<0.39.0,>=0.37.2` constraint) has advisories fixed only in
  0.40.0+ — not upgradable without also upgrading FastAPI itself, which
  is a larger, separately-risky change this phase's "do not redesign
  the application" scope doesn't cover. **Accepted risk, documented,
  not fixed.**
- **`mlflow` 2.16.2** has advisories fixed only in 2.17.0+ — a version
  bump with likely breaking changes across the whole training/registry/
  inference pipeline every prior phase built and tested against.
  **Accepted risk, documented, not fixed** — recommend evaluating an
  MLflow upgrade as its own dedicated phase, with the full test suite
  re-run against it, rather than folding it into a deployment phase.
- **`pyarrow`** — transitive via `mlflow`; tied to the same upgrade
  decision above.
- **`pytest`** — dev/CI only as of this phase's `requirements.txt` /
  `requirements-dev.txt` split (see below); never present in the
  production image, so this is a much lower-severity finding than the
  three above.

**Dev/prod dependency split (new this phase)**: `backend/requirements.txt`
now lists runtime dependencies only; `pytest`/`ruff`/`pip-audit` moved to
new `backend/requirements-dev.txt`. `backend/Dockerfile` already only
ever installed `requirements.txt` (unchanged), so this means test/lint
tooling — and whatever CVEs it carries — **no longer ships inside the
production image at all**, without any Dockerfile change required.

### Cost considerations

**Approximate only** — Render's pricing changes over time and this
project's own attempts to fetch its live pricing tables returned mostly
JavaScript-rendered content this tooling couldn't read; the numbers
below come from Render's own documentation pages that *did* return
plain text (fetched while writing this section — see sources) plus
independent aggregator sites, not a single authoritative price sheet.
**Check [render.com/pricing](https://render.com/pricing) directly for
current numbers before budgeting.**

- Render's **free tier** cannot actually run this deployment
  persistently: free Postgres **expires 30 days after creation**, free
  web services **spin down after 15 minutes of inactivity** (cold
  starts) and can't attach a persistent disk, and free Key Value has no
  persistence at all — confirmed directly from Render's docs. Fine for
  a brief demo, not for "real accessible production-style application."
- A minimal **paid** deployment (`render.yaml`'s exact plan choices —
  `0.5c-512mb` for the 4 Docker services, `0.1c-256mb`/1GB disk for
  Postgres, `256mb` for Key Value, 5GB MLflow disk) is roughly in the
  **$25–45/month** range based on the figures found — small
  service/worker plans reportedly start around $7/mo each (×4 =
  ~$28/mo), the smallest paid Postgres plan around $6–7/mo, the
  smallest paid Key Value plan around $10/mo, plus ~$0.25–0.30/GB/month
  for disk storage (a few GB here is a few dollars). This is a rough
  order-of-magnitude, not a quote — Render's actual current per-plan
  pricing should be confirmed on their pricing page for the specific
  plan codes in `render.yaml` before committing to this.
- An S3-compatible bucket (AWS S3 or Cloudflare R2) for dataset storage
  adds a small, usage-based cost on top — trivial at this project's
  data volumes (a handful of CSVs), typically cents/month.

Sources: [Render Blueprint YAML Reference](https://render.com/docs/blueprint-spec), [Render free tier docs](https://render.com/docs/free), [Render Docker docs](https://render.com/docs/docker), [Render Key Value docs](https://render.com/docs/key-value), and general pricing aggregation via web search (render.com/pricing itself did not return readable pricing tables to this tooling).

### Real deployment verification checklist

Not run — no cloud credentials were available in the environment this
phase was built in (confirmed: no `docker`, `aws`, `gcloud`, `az`,
`flyctl`, `railway`, `doctl`, `heroku`, or `terraform` CLI present). If
you deploy this yourself, this is the checklist to work through (it
maps directly onto this phase's own requirements):

- [ ] Open the production frontend URL over HTTPS
- [ ] Log in with the admin user created via `scripts/create_admin.py`
- [ ] Confirm an authenticated API call succeeds (e.g. `/models`) and
      an unauthenticated one is rejected (open the frontend in a
      private/incognito window, confirm it redirects to `/login`)
- [ ] Submit a real prediction (`/predict` via the dashboard)
- [ ] List models (`/models`) and confirm the production model shows
- [ ] Check `/monitoring` renders real data
- [ ] Submit an async training job and confirm `predictiq-worker`'s
      Render logs show it being picked up and completed
- [ ] Confirm the resulting run appears in MLflow (reachable only
      privately — check via a Render Shell on `predictiq-backend`:
      `curl http://predictiq-mlflow:5001/health`, or temporarily check
      `predictiq-mlflow`'s logs)
- [ ] Restart `predictiq-postgres` (or trigger a redeploy) and confirm
      existing models/predictions/users are still there
- [ ] Restart `predictiq-mlflow` and confirm previously logged runs and
      artifacts are still there (proves the disk actually persisted,
      not just that the container came back up)
- [ ] Confirm HTTPS is enforced (Render does this automatically for
      every `web` service — verify by requesting the `http://` URL and
      confirming a redirect)

### Verification status

Exactly as this phase's brief requires — distinguishing what was
**VERIFIED LOCALLY**, **VERIFIED IN CLOUD**, **STATICALLY VALIDATED**,
or **NOT VERIFIED**, with no gap hidden:

| Item | Status |
|---|---|
| Backend: 215 tests, `ruff check .`, Alembic migrations (incl. the 2 new Phase 10 ones) | **VERIFIED LOCALLY** (real Postgres/Redis on this machine) |
| Frontend: 35 tests, `eslint`, `tsc --noEmit`, `next build` | **VERIFIED LOCALLY** |
| `docker-compose.yml`, `render.yaml`, `.github/workflows/ci.yml` YAML syntax | **STATICALLY VALIDATED** (parsed with PyYAML) |
| `render.yaml` schema correctness (field names, `fromService`/`fromDatabase` syntax, `ipAllowList`, plan enum values, `dockerCommand` vs `startCommand`) | **STATICALLY VALIDATED** against Render's own published Blueprint documentation, fetched and cross-checked while writing this file — not validated by Render's own tooling (no account to run a real Blueprint sync against) |
| `dataset_storage.py`'s S3 backend | **VERIFIED LOCALLY** against a fake in-memory S3 client (`tests/unit/test_dataset_storage.py`) — **NOT VERIFIED** against real AWS S3/Cloudflare R2 |
| MLflow with an S3 `--default-artifact-root` | **NOT VERIFIED** — no real S3/R2 credentials available; this relies on MLflow's own documented S3ArtifactRepository support, not custom code in this project |
| Docker image builds (backend, frontend, mlflow) | **NOT VERIFIED** in this environment (no Docker install — see [Docker setup](#docker-setup)'s own note); will run for real in CI's `docker-build` job the next time this repo is pushed to GitHub, but that push has not happened from this environment |
| `.github/workflows/ci.yml` actually executing on GitHub Actions | **NOT VERIFIED** — no push was made; syntax-validated only |
| Render deployment (any service actually created, any URL actually reachable) | **NOT VERIFIED** — no Render account/credentials available |
| Everything in [Real deployment verification checklist](#real-deployment-verification-checklist) | **NOT VERIFIED** |

## Testing

**Backend**: 215 tests (`cd backend && pytest`), run against a real local
PostgreSQL test database and, where the test specifically exercises it, a
real Redis connection — not mocked DB/queue behavior for anything
verifying constraints or transactions. This includes Phase 9's real
login/logout flow (`tests/integration/test_auth_api.py`, run against a
real Redis-backed session store): successful/failed/rate-limited login,
password hashing (`tests/unit/test_security.py`), session
creation/invalidation, unauthenticated rejection across every protected
router, non-admin rejection of promote/rollback, and the audit trail
recording the acting admin's email — every existing (pre-Phase-9) test
still runs through the same real login flow rather than a mocked auth
dependency, via `tests/conftest.py`'s `client` fixture. Phase 10 adds
`tests/unit/test_dataset_storage.py` (local backend behavior via the
real filesystem; the S3 backend against an in-memory fake client — see
[Production deployment](#production-deployment) for why real S3/R2
wasn't available to test against).

**Frontend**: 35 tests across 10 files (`cd frontend && npm test`)
covering model list rendering, model detail + lineage, training job
submission, the prediction table (including that raw features never leak
into the list view), monitoring's insufficient-data/ok/no-production-model
states, API-client error handling (HTTP errors, malformed JSON, network
failure, timeout), and Phase 9's login form (success, failure message,
disabled-while-submitting), route-guard redirects (`AppShell`), and
admin-only UI gating (`PromotionControls`).

Both suites are lint-clean (`ruff check .` / `npm run lint`) and the
frontend has a clean `npm run build`.

## Deployment status

Phase 8 added Docker/Docker Compose packaging: `backend/Dockerfile`,
`frontend/Dockerfile`, `infra/mlflow/Dockerfile`,
`infra/postgres/init-multi-db.sh`, and the root `docker-compose.yml` —
see [Docker setup](#docker-setup) for the full architecture and how it
maps onto the native setup this project was originally developed
against. Phase 10 added a production cloud deployment configuration on
top of that same Docker architecture (`render.yaml`,
`.github/workflows/ci.yml`) — see
[Production deployment](#production-deployment) for the full picture,
including its own honest verification-status breakdown (nothing there
has actually been deployed to a real cloud account).

**Verification honesty note**: the environment this Phase was built in
does not have Docker installed, so the Compose stack has been
*statically* validated only — YAML syntax, `Settings`/env-var wiring
traced through the actual source (`app/core/config.py`,
`app/core/mlflow_config.py`, `db/session.py`,
`workers/celery_app.py`), and the `frontend`'s `output: "standalone"`
build (the same code path the Docker image builds) run for real via
`npm run build`. It has **not** been built or run as containers, so
claims like "the stack starts cleanly," "a training job completes
inside the containers," or "the worker reaches MLflow over the Compose
network" are not yet backed by an actual run. Run the commands in
[Docker setup](#docker-setup) locally (Docker Desktop/Colima, etc.) to
get that end-to-end confirmation — see this phase's final report for the
exact list of what was and wasn't verified.

Everything else described in this README **is** real and has been run
end-to-end locally: real Postgres, real Redis, a real MLflow tracking
server, a real Celery worker, and the real Next.js dashboard against the
real FastAPI backend (all via the native setup).

## Limitations

- **Docker Compose stack not yet runtime-verified** (see
  [Deployment status](#deployment-status)) — authored and statically
  validated, not yet built/run as containers in this environment.
- **Authentication is real but deliberately minimal** (Phase 9 — see
  [Authentication & authorization](#authentication--authorization)).
  Explicitly **not** implemented, by scope decision rather than
  oversight:
  - **No MFA/2FA.**
  - **No SSO/OAuth** (Google/GitHub/SAML/etc) — email+password only.
  - **No advanced RBAC** — exactly two levels (user, administrator), no
    per-resource permissions, no roles table.
  - **No account recovery / "forgot password" flow** — a forgotten
    password is reset by re-running
    `scripts/create_admin.py` for that email.
  - **No email verification** — there's no email-sending
    infrastructure in this project at all; an email is only ever used
    as a login identifier.
  - **No self-service registration** — accounts are provisioned via
    `scripts/create_admin.py` only.
  - **No per-user data isolation** — the `predictions`/`jobs`/etc.
    tables have no owning-user column; every authenticated user (and
    every admin) sees the same shared data. Adding real multi-tenant
    isolation would be a schema change, not just an API change.
  - **`SECRET_KEY` defaults to a checked-in placeholder**
    (`dev-insecure-secret-key-change-me`) — safe for local dev, **must**
    be overridden (see the Environment variables table) for anything
    reachable outside your own machine.
  - **`SESSION_COOKIE_SECURE` defaults to `false`** — required for local
    plain-HTTP dev; **must** be `true` wherever this is actually served
    over HTTPS, or a session cookie could be sent over an unencrypted
    connection.
  - **Login rate limiting trusts `request.client.host` directly by
    default** — correct for native dev and Docker Compose, where
    there's no reverse proxy in front of the backend. Phase 10's Render
    deployment sits behind Render's own edge proxy for every request,
    which would otherwise collapse the rate limiter to one shared
    bucket keyed on the proxy's address instead of each real client —
    `TRUST_PROXY_HEADERS=true` (set only in `render.yaml`, not the
    native/Compose defaults) switches to trusting `X-Forwarded-For`'s
    left-most entry instead, safe specifically because Render's
    topology makes the backend unreachable except via that proxy. See
    `app/routers/auth.py`'s `_client_identifier`.
  - **A session that expires mid-page doesn't auto-redirect to
    /login** — the frontend's route guard
    (`frontend/src/components/auth/AppShell.tsx`) checks auth state on
    navigation/mount, not on every subsequent API call; a call that 401s
    after the session already expired surfaces through that page's
    existing error state (`role="alert"`, with retry) rather than an
    automatic redirect. The backend still correctly rejects the request
    either way — this is a UX gap, not a security one.
- **Prometheus counters are per-process** — a separate Celery worker
  process's prediction counts don't appear in the API process's
  `/metrics`. The dashboard labels this explicitly rather than hiding it;
  real cross-process aggregation would need `prometheus_client`'s
  multiprocess mode or a Pushgateway.
- **Multi-worker model-cache staleness** — with more than one API worker
  process, a promotion is only guaranteed visible to every worker after
  each has served one more request (no cross-process push invalidation).
- **macOS + Celery prefork**: Celery's default worker pool can segfault
  on macOS when forking a process that already loaded XGBoost/numpy/
  libomp; local dev uses `--pool=solo`. This is a macOS-specific
  development quirk, not a deployment concern for Linux containers.
- **Drift reference recomputed per request** — the reference distribution
  is read from the training CSV on each `GET /monitoring/drift` call
  rather than cached; fine at this project's scale, would need caching
  under real load.

## Future improvements

- Runtime-verify the Docker Compose stack end-to-end on a machine with
  Docker installed (see [Deployment status](#deployment-status)), then
  update this README to reflect a real run instead of static validation.
- MFA, SSO, self-service registration, account recovery, and real
  per-resource RBAC — all explicitly out of scope for Phase 9 (see
  [Limitations](#limitations)); add only if this leaves a local/demo
  context.
- Per-user data isolation (an owning-user column on
  predictions/jobs/etc.) if this stops being single-tenant.
- Cross-process Prometheus aggregation (multiprocess mode or a
  Pushgateway) so worker-side prediction counts are visible from `/metrics`.
- Cache the drift reference distribution per model version instead of
  recomputing it from the CSV on every request.
- Actually deploy to Render and work through the
  [Real deployment verification checklist](#real-deployment-verification-checklist)
  — Phase 10 produced a statically-validated `render.yaml` and CI
  pipeline, not a live deployment (see
  [Verification status](#verification-status)); this is the single
  biggest gap between what's built and what's proven.
- Push this repo to GitHub and confirm `.github/workflows/ci.yml`
  actually passes on real GitHub Actions infrastructure — authored and
  YAML-validated, not yet executed.
- Verify `dataset_storage.py`'s S3 backend and MLflow's S3 artifact root
  against a real bucket (AWS S3 or Cloudflare R2) — currently verified
  only against a fake in-memory S3 client (see
  [Production deployment](#production-deployment)).
- Upgrade `mlflow` (2.16.2 → 2.17+) and `fastapi`/`starlette` to clear
  the dependency-vulnerability findings in
  [Security review](#security-review) that a version bump alone can't
  safely fix within this phase's scope — needs its own dedicated phase
  with the full test suite re-verified against the new versions.

## Interview talking points

- **Why a shared `DatasetSchema` module** instead of separate column
  lists in ingestion, training, and inference: one source of truth that
  can't silently drift between the three stages (backend/ml/schema.py).
- **Why the exact same fitted Pipeline object serves both training and
  inference**: no second hand-maintained preprocessing implementation at
  serving time.
- **Why partial unique indexes, not just application checks**, enforce
  "one production model": a race between two concurrent promotion
  requests is caught by Postgres itself, and this was validated with a
  concurrency test using a `threading.Barrier` to force a genuine race
  (a naive two-thread test turned out to run sequentially by accident on
  fast local Postgres).
- **Why KS for numeric and PSI for categorical drift**, not one test for
  everything — see `docs/monitoring.md`.
- **Two real "process-global config" bugs found and fixed during manual
  verification**, not caught by unit tests alone: MLflow's tracking URI
  defaulting to a local file store in both the API process and the Celery
  worker (fixed by centralizing configuration in
  `app/core/mlflow_config.py`, with a regression test), and Celery's
  `worker_hijack_root_logger` silently discarding structured JSON logs
  inside the worker.
- **A real frontend bug found via manual browser verification, not code
  review**: pages checked `useQuery`'s `isLoading` (only true while
  actively fetching) instead of `isPending` (true for the whole pending
  status, including a retry's backoff pause) before deciding whether to
  show a loading state — a transient in-between-retries render could fall
  through to an error branch with `error` still `null`, showing a
  hardcoded fallback message instead of the real backend error. Fixed
  across all seven pages, with a regression test.
