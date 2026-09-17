"""Model evaluation metrics.

Reports precision/recall/F1/ROC-AUC/confusion-matrix — accuracy is
included but never treated as the headline number, since the churn
dataset is imbalanced (~26.5% positive) and accuracy alone rewards
always-predicting-the-majority-class.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def flatten_metrics_for_mlflow(metrics: dict, prefix: str) -> dict[str, float]:
    """MLflow metrics must be scalar floats — flatten the confusion matrix."""
    flat = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            for cm_key, cm_value in value.items():
                flat[f"{prefix}_{key}_{cm_key}"] = float(cm_value)
        else:
            flat[f"{prefix}_{key}"] = float(value)
    return flat
