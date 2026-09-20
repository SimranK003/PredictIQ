"""Reproducible training pipeline: dataset -> preprocessing -> feature
engineering -> train/val/test split -> model training -> evaluation ->
MLflow experiment -> model artifact.

Usage:
    python -m ml.training.train --dataset-id <uuid> [--config path.yaml]

The pure ML logic (run_training_pipeline) is kept free of any MLflow
dependency so it can be unit-tested without a tracking server. MLflow
logging is a separate step (log_training_run_to_mlflow) applied to each
result afterward.
"""

import argparse
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.core.logging import configure_logging, get_logger
from app.core.mlflow_config import configure_mlflow
from app.services.dataset_storage import read_dataset_dataframe
from app.services.registry import register_candidate
from db.models import Dataset, ModelVersionRecord
from db.session import SessionLocal
from ml.preprocessing import build_full_pipeline
from ml.schema import CHURN_SCHEMA, DatasetSchema
from ml.training.config import TrainingConfig, load_training_config
from ml.training.data import prepare_features_and_target, split_dataset
from ml.training.evaluate import evaluate_predictions, flatten_metrics_for_mlflow
from ml.training.models import ALGORITHMS, build_classifier, hyperparams_for

logger = get_logger(__name__)

PREPROCESSING_DESCRIPTION = (
    "numeric: median-impute + standard-scale; categorical: most-frequent-impute + one-hot"
)


@dataclass
class TrainingRunResult:
    algorithm: str
    pipeline: Pipeline
    hyperparams: dict
    val_metrics: dict
    test_metrics: dict
    n_train: int
    n_val: int
    n_test: int


@dataclass
class MlflowRunHandle:
    run_id: str
    experiment_id: str
    artifact_uri: str


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_training_pipeline(
    df: pd.DataFrame,
    schema: DatasetSchema,
    config: TrainingConfig,
    *,
    algorithms: Sequence[str] = ALGORITHMS,
) -> list[TrainingRunResult]:
    """Train and evaluate the requested algorithms (all three by default).
    No MLflow calls here — this function is what unit/integration tests
    exercise directly.
    """
    unknown = set(algorithms) - set(ALGORITHMS)
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {sorted(unknown)}. Valid: {ALGORITHMS}")

    x, y = prepare_features_and_target(df, schema)
    split = split_dataset(
        x, y, test_size=config.test_size, val_size=config.val_size, random_state=config.random_state
    )

    n_neg = int((split.y_train == 0).sum())
    n_pos = int((split.y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    results: list[TrainingRunResult] = []
    for algorithm in algorithms:
        classifier = build_classifier(algorithm, config, scale_pos_weight)
        pipeline = build_full_pipeline(schema, classifier)
        pipeline.fit(split.x_train, split.y_train)

        val_pred = pipeline.predict(split.x_val)
        val_proba = pipeline.predict_proba(split.x_val)[:, 1]
        val_metrics = evaluate_predictions(split.y_val.to_numpy(), val_pred, val_proba)

        test_pred = pipeline.predict(split.x_test)
        test_proba = pipeline.predict_proba(split.x_test)[:, 1]
        test_metrics = evaluate_predictions(split.y_test.to_numpy(), test_pred, test_proba)

        results.append(
            TrainingRunResult(
                algorithm=algorithm,
                pipeline=pipeline,
                hyperparams=hyperparams_for(algorithm, config, scale_pos_weight),
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                n_train=len(split.x_train),
                n_val=len(split.x_val),
                n_test=len(split.x_test),
            )
        )

    return results


def log_training_run_to_mlflow(
    result: TrainingRunResult,
    *,
    dataset: Dataset,
    schema: DatasetSchema,
    config: TrainingConfig,
    sample_input: pd.DataFrame,
    training_job_id: uuid.UUID | None = None,
) -> MlflowRunHandle:
    """Log one trained pipeline as an MLflow run."""
    with mlflow.start_run(run_name=result.algorithm) as run:
        tags = {
            "algorithm": result.algorithm,
            "dataset_id": str(dataset.id),
            "dataset_content_hash": dataset.content_hash,
            "dataset_filename": dataset.filename,
            "schema_name": schema.name,
            "git_commit": _git_commit(),
            "training_timestamp": datetime.now(UTC).isoformat(),
        }
        if training_job_id is not None:
            tags["training_job_id"] = str(training_job_id)
        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                **result.hyperparams,
                "test_size": config.test_size,
                "val_size": config.val_size,
                "random_state": config.random_state,
                "n_train_rows": result.n_train,
                "n_val_rows": result.n_val,
                "n_test_rows": result.n_test,
                "preprocessing": PREPROCESSING_DESCRIPTION,
            }
        )
        mlflow.log_metrics(flatten_metrics_for_mlflow(result.val_metrics, "val"))
        mlflow.log_metrics(flatten_metrics_for_mlflow(result.test_metrics, "test"))
        mlflow.sklearn.log_model(
            result.pipeline, artifact_path="model", input_example=sample_input
        )
        return MlflowRunHandle(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            artifact_uri=f"runs:/{run.info.run_id}/model",
        )


def _load_dataset(dataset_id: str) -> Dataset:
    db = SessionLocal()
    try:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise SystemExit(f"Dataset {dataset_id} not found.")
        if not dataset.is_valid:
            raise SystemExit(
                f"Dataset {dataset_id} failed validation at ingestion time and cannot be "
                "used for training. Check its quality_report via GET /datasets/{id}."
            )
        return dataset
    finally:
        db.close()


def train_and_register(
    db: Session,
    *,
    dataset: Dataset,
    config: TrainingConfig,
    algorithms: Sequence[str] = ALGORITHMS,
    schema: DatasetSchema = CHURN_SCHEMA,
    training_job_id: uuid.UUID | None = None,
) -> list[ModelVersionRecord]:
    """Full pipeline: load raw data -> train -> evaluate -> log to MLflow
    -> register each result as a candidate.

    The single implementation of "how training becomes registered
    candidates" — called by both the CLI (python -m ml.training.train)
    and the async training Celery task (app/services/training.py), so
    there is exactly one place this orchestration logic lives.
    """
    configure_mlflow()

    df = read_dataset_dataframe(dataset.storage_path)
    results = run_training_pipeline(df, schema, config, algorithms=algorithms)

    sample_input = df[list(schema.all_feature_columns)].head(3)
    git_commit = _git_commit()

    candidates: list[ModelVersionRecord] = []
    for result in results:
        run_handle = log_training_run_to_mlflow(
            result,
            dataset=dataset,
            schema=schema,
            config=config,
            sample_input=sample_input,
            training_job_id=training_job_id,
        )
        candidate = register_candidate(
            db,
            dataset=dataset,
            algorithm=result.algorithm,
            mlflow_run_id=run_handle.run_id,
            mlflow_experiment_id=run_handle.experiment_id,
            artifact_uri=run_handle.artifact_uri,
            metrics=result.test_metrics,
            params={
                **result.hyperparams,
                "test_size": config.test_size,
                "val_size": config.val_size,
                "random_state": config.random_state,
                "preprocessing": PREPROCESSING_DESCRIPTION,
            },
            git_commit=git_commit,
            training_job_id=training_job_id,
        )
        logger.info(
            "model_candidate_registered",
            extra={
                "dataset_id": str(dataset.id),
                "training_job_id": str(training_job_id) if training_job_id else None,
                "mlflow_run_id": run_handle.run_id,
                "model_version_id": str(candidate.id),
                "algorithm": result.algorithm,
                "test_roc_auc": result.test_metrics["roc_auc"],
            },
        )
        candidates.append(candidate)

    return candidates


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train churn prediction baseline models.")
    parser.add_argument("--dataset-id", required=True, help="UUID of an ingested, valid dataset.")
    parser.add_argument("--config", default=None, help="Path to a training config YAML file.")
    args = parser.parse_args(argv)

    dataset = _load_dataset(args.dataset_id)
    config = load_training_config(args.config)

    logger.info(
        "training_started",
        extra={
            "dataset_id": str(dataset.id),
            "n_rows": dataset.n_rows,
            "config": config.resolved_dict(),
        },
    )

    print("\n" + "=" * 88)
    print(f"Training run — dataset {dataset.id} ({dataset.filename}, {dataset.n_rows} rows)")
    print("=" * 88)

    db = SessionLocal()
    try:
        candidates = train_and_register(db, dataset=dataset, config=config)
        for candidate in candidates:
            tm = candidate.metrics
            cm = tm["confusion_matrix"]
            print(
                f"\n[{candidate.algorithm}]  mlflow_run_id={candidate.mlflow_run_id}  "
                f"registered as {candidate.version_label} (candidate, id={candidate.id})"
            )
            print(
                f"  test  -> precision={tm['precision']:.4f} recall={tm['recall']:.4f} "
                f"f1={tm['f1']:.4f} roc_auc={tm['roc_auc']:.4f} accuracy={tm['accuracy']:.4f}"
            )
            print(
                f"  confusion_matrix -> TN={cm['true_negative']} FP={cm['false_positive']} "
                f"FN={cm['false_negative']} TP={cm['true_positive']}"
            )

        print("\n" + "=" * 88)
        best = max(candidates, key=lambda c: c.metrics["roc_auc"])
        print(f"Best by test ROC-AUC: {best.algorithm} ({best.metrics['roc_auc']:.4f})")
        print("All results registered as candidates. Promote explicitly via POST /models/promote.")
        print("=" * 88 + "\n")
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
