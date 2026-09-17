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
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from db.models import Dataset
from db.session import SessionLocal
from ml.preprocessing import build_full_pipeline
from ml.schema import CHURN_SCHEMA, DatasetSchema
from ml.training.config import TrainingConfig, load_training_config
from ml.training.data import load_raw_dataframe, prepare_features_and_target, split_dataset
from ml.training.evaluate import evaluate_predictions, flatten_metrics_for_mlflow
from ml.training.models import ALGORITHMS, build_classifier, hyperparams_for

logger = get_logger(__name__)


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
    df: pd.DataFrame, schema: DatasetSchema, config: TrainingConfig
) -> list[TrainingRunResult]:
    """Train and evaluate all baseline algorithms. No MLflow calls here —
    this function is what unit/integration tests exercise directly.
    """
    x, y = prepare_features_and_target(df, schema)
    split = split_dataset(
        x, y, test_size=config.test_size, val_size=config.val_size, random_state=config.random_state
    )

    n_neg = int((split.y_train == 0).sum())
    n_pos = int((split.y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    results: list[TrainingRunResult] = []
    for algorithm in ALGORITHMS:
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
) -> str:
    """Log one trained pipeline as an MLflow run. Returns the run_id."""
    with mlflow.start_run(run_name=result.algorithm) as run:
        mlflow.set_tags(
            {
                "algorithm": result.algorithm,
                "dataset_id": str(dataset.id),
                "dataset_content_hash": dataset.content_hash,
                "dataset_filename": dataset.filename,
                "schema_name": schema.name,
                "git_commit": _git_commit(),
                "training_timestamp": datetime.now(UTC).isoformat(),
            }
        )
        mlflow.log_params(
            {
                **result.hyperparams,
                "test_size": config.test_size,
                "val_size": config.val_size,
                "random_state": config.random_state,
                "n_train_rows": result.n_train,
                "n_val_rows": result.n_val,
                "n_test_rows": result.n_test,
            }
        )
        mlflow.log_metrics(flatten_metrics_for_mlflow(result.val_metrics, "val"))
        mlflow.log_metrics(flatten_metrics_for_mlflow(result.test_metrics, "test"))
        mlflow.sklearn.log_model(
            result.pipeline, artifact_path="model", input_example=sample_input
        )
        return run.info.run_id


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


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train churn prediction baseline models.")
    parser.add_argument("--dataset-id", required=True, help="UUID of an ingested, valid dataset.")
    parser.add_argument("--config", default=None, help="Path to a training config YAML file.")
    args = parser.parse_args(argv)

    dataset = _load_dataset(args.dataset_id)
    schema = CHURN_SCHEMA
    config = load_training_config(args.config)

    logger.info(
        "training_started",
        extra={
            "dataset_id": str(dataset.id),
            "n_rows": dataset.n_rows,
            "config": config.resolved_dict(),
        },
    )

    df = load_raw_dataframe(dataset.storage_path)
    results = run_training_pipeline(df, schema, config)

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    sample_input = df[list(schema.all_feature_columns)].head(3)

    print("\n" + "=" * 88)
    print(f"Training run — dataset {dataset.id} ({dataset.filename}, {dataset.n_rows} rows)")
    print("=" * 88)

    for result in results:
        run_id = log_training_run_to_mlflow(
            result, dataset=dataset, schema=schema, config=config, sample_input=sample_input
        )
        tm = result.test_metrics
        cm = tm["confusion_matrix"]
        print(f"\n[{result.algorithm}]  mlflow_run_id={run_id}")
        print(f"  hyperparams: {result.hyperparams}")
        print(
            f"  test  -> precision={tm['precision']:.4f} recall={tm['recall']:.4f} "
            f"f1={tm['f1']:.4f} roc_auc={tm['roc_auc']:.4f} accuracy={tm['accuracy']:.4f}"
        )
        print(
            f"  confusion_matrix -> TN={cm['true_negative']} FP={cm['false_positive']} "
            f"FN={cm['false_negative']} TP={cm['true_positive']}"
        )

    print("\n" + "=" * 88)
    best = max(results, key=lambda r: r.test_metrics["roc_auc"])
    print(f"Best by test ROC-AUC: {best.algorithm} ({best.test_metrics['roc_auc']:.4f})")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    sys.exit(main())
