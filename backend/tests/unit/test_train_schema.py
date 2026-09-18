"""Guards against TrainRequest's hyperparameter fields silently drifting
from ml.training.config.TrainingConfig — if someone adds a hyperparameter
to one without the other, this fails loudly instead of the two quietly
diverging (POST /train would then be unable to configure a field the
CLI/Celery path supports, or vice versa).
"""

from app.schemas.train import TrainRequest
from ml.training.config import TrainingConfig


def test_train_request_hyperparameter_fields_match_training_config():
    non_hyperparam_fields = {"dataset_id", "algorithms"}
    request_fields = set(TrainRequest.model_fields.keys()) - non_hyperparam_fields
    config_fields = set(TrainingConfig.model_fields.keys())

    assert request_fields == config_fields


def test_resolved_config_overrides_only_includes_explicitly_set_fields():
    request = TrainRequest(dataset_id="11111111-1111-1111-1111-111111111111", random_state=99)
    overrides = request.resolved_config_overrides()

    assert overrides == {"random_state": 99}


def test_resolved_config_overrides_produces_a_valid_training_config():
    request = TrainRequest(
        dataset_id="11111111-1111-1111-1111-111111111111",
        random_forest_n_estimators=50,
        xgboost_max_depth=3,
    )
    config = TrainingConfig(**request.resolved_config_overrides())

    assert config.random_forest_n_estimators == 50
    assert config.xgboost_max_depth == 3
    assert config.test_size == 0.15  # untouched fields keep their default
