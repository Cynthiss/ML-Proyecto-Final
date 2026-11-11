"""Pruebas de los nodos de ``model_training`` con datos sintéticos pequeños."""

import numpy as np
import pandas as pd
import pytest

from sales_performance.pipelines.model_training.nodes import (
    _time_folds,
    get_feature_columns,
    train_models,
    tune_hyperparameters,
)

PARAMS = {
    "target": "Revenue",
    "exclude_features": ["SalesPersonID", "Date"],
    "cv_splits": 2,
    "random_state": 0,
    "optuna": {
        "n_trials": 2,
        "lightgbm": {
            "n_estimators": [10, 20],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [4, 8],
            "min_child_samples": [5, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.01, 1.0],
        },
        "ridge": {"alpha": [0.1, 10.0]},
    },
}


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        [(s, d) for s in [1, 2, 3] for d in dates], columns=["SalesPersonID", "Date"]
    )
    df["x1"] = rng.normal(size=len(df))
    df["Revenue"] = 100 + 10 * df["x1"] + rng.normal(size=len(df))
    return df


def test_feature_columns_exclude_keys_and_target(train):
    assert get_feature_columns(train, PARAMS) == ["x1"]


def test_time_folds_never_validate_on_past(train):
    for train_idx, val_idx in _time_folds(train["Date"], 3):
        assert train["Date"].iloc[train_idx].max() < train["Date"].iloc[val_idx].min()


def test_tune_and_train(train):
    best, trials = tune_hyperparameters(train, PARAMS)
    assert set(best) == {"ridge", "lightgbm"}
    assert len(trials) == 2 * PARAMS["optuna"]["n_trials"]
    ridge, lightgbm = train_models(train, best, PARAMS)
    x = train[["x1"]]
    # la relación es lineal y fuerte: Ridge debe capturarla casi perfecta
    assert np.corrcoef(ridge.predict(x), train["Revenue"])[0, 1] > 0.95
    assert len(lightgbm.predict(x)) == len(train)
