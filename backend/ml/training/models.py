"""Model builders for the three baseline algorithms.

Class imbalance (~26.5% positive in the churn dataset) is handled per
algorithm rather than by resampling (e.g. SMOTE): class_weight="balanced"
for LogisticRegression/RandomForest reweights the loss without
synthesizing rows, and XGBoost's scale_pos_weight does the equivalent.
This keeps the pipeline simple and avoids resampling logic that would
need to be carefully scoped to the training fold only.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.training.config import TrainingConfig


def build_logistic_regression(config: TrainingConfig) -> LogisticRegression:
    return LogisticRegression(
        C=config.logistic_regression_c,
        max_iter=config.logistic_regression_max_iter,
        class_weight="balanced",
        random_state=config.random_state,
    )


def build_random_forest(config: TrainingConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.random_forest_n_estimators,
        max_depth=config.random_forest_max_depth,
        min_samples_leaf=config.random_forest_min_samples_leaf,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )


def build_xgboost(config: TrainingConfig, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config.xgboost_n_estimators,
        max_depth=config.xgboost_max_depth,
        learning_rate=config.xgboost_learning_rate,
        subsample=config.xgboost_subsample,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=-1,
    )


def hyperparams_for(algorithm: str, config: TrainingConfig, scale_pos_weight: float) -> dict:
    if algorithm == "logistic_regression":
        return {
            "C": config.logistic_regression_c,
            "max_iter": config.logistic_regression_max_iter,
            "class_weight": "balanced",
        }
    if algorithm == "random_forest":
        return {
            "n_estimators": config.random_forest_n_estimators,
            "max_depth": config.random_forest_max_depth,
            "min_samples_leaf": config.random_forest_min_samples_leaf,
            "class_weight": "balanced",
        }
    if algorithm == "xgboost":
        return {
            "n_estimators": config.xgboost_n_estimators,
            "max_depth": config.xgboost_max_depth,
            "learning_rate": config.xgboost_learning_rate,
            "subsample": config.xgboost_subsample,
            "scale_pos_weight": round(scale_pos_weight, 4),
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


ALGORITHMS = ("logistic_regression", "random_forest", "xgboost")


def build_classifier(algorithm: str, config: TrainingConfig, scale_pos_weight: float):
    if algorithm == "logistic_regression":
        return build_logistic_regression(config)
    if algorithm == "random_forest":
        return build_random_forest(config)
    if algorithm == "xgboost":
        return build_xgboost(config, scale_pos_weight)
    raise ValueError(f"Unknown algorithm: {algorithm}")
