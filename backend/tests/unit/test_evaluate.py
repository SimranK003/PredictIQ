import numpy as np

from ml.training.evaluate import evaluate_predictions, flatten_metrics_for_mlflow


def test_evaluate_predictions_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.95])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_evaluate_predictions_known_confusion_matrix():
    # 1 TN, 1 FP, 1 FN, 1 TP
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.6, 0.4, 0.8])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_evaluate_predictions_handles_no_positive_predictions_without_error():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_flatten_metrics_for_mlflow_expands_confusion_matrix():
    metrics = {
        "accuracy": 0.9,
        "confusion_matrix": {
            "true_negative": 5,
            "false_positive": 1,
            "false_negative": 2,
            "true_positive": 3,
        },
    }
    flat = flatten_metrics_for_mlflow(metrics, "test")

    assert flat["test_accuracy"] == 0.9
    assert flat["test_confusion_matrix_true_negative"] == 5.0
    assert flat["test_confusion_matrix_true_positive"] == 3.0
    assert all(isinstance(v, float) for v in flat.values())
