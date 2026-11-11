"""Nodos del pipeline ``model_training``.

Se comparan dos candidatos con objetivos distintos:

- **Ridge**: regresión lineal regularizada, fácil de interpretar y de implementar.
- **LightGBM**: gradient boosting, capta relaciones no lineales e interacciones.

Ambos se optimizan con Optuna minimizando el MAE en validación cruzada temporal
(cada fold entrena con fechas anteriores y valida con fechas posteriores).
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame, params: dict[str, Any]) -> list[str]:
    """Devuelve las columnas que se usan como variables explicativas.

    Args:
        df: Tabla de modelado.
        params: Parámetros con ``target`` y ``exclude_features``.

    Returns:
        Lista de nombres de columnas.
    """
    excluded = {params["target"], *params["exclude_features"]}
    return [c for c in df.columns if c not in excluded]


def _time_folds(dates: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera folds temporales por bloques de fechas (sin mezclar días).

    Args:
        dates: Fecha de cada fila.
        n_splits: Número de folds.

    Returns:
        Lista de pares (índices de entrenamiento, índices de validación).
    """
    unique_dates = np.sort(dates.unique())
    folds = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        train_mask = dates.isin(unique_dates[train_idx]).to_numpy()
        val_mask = dates.isin(unique_dates[val_idx]).to_numpy()
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


def build_ridge(alpha: float) -> Pipeline:
    """Crea el modelo Ridge con estandarización de variables."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def build_lightgbm(hyperparams: dict[str, Any], random_state: int) -> lgb.LGBMRegressor:
    """Crea el modelo LightGBM con los hiperparámetros dados."""
    return lgb.LGBMRegressor(
        **hyperparams,
        subsample_freq=1,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )


def _cv_mae(model_factory, x: pd.DataFrame, y: pd.Series, folds) -> float:
    """MAE promedio de validación cruzada temporal."""
    scores = []
    for train_idx, val_idx in folds:
        model = model_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        scores.append(
            mean_absolute_error(y.iloc[val_idx], model.predict(x.iloc[val_idx]))
        )
    return float(np.mean(scores))


def _suggest_lightgbm(trial: optuna.Trial, space: dict[str, list]) -> dict[str, Any]:
    """Define el espacio de búsqueda de LightGBM para un trial de Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *space["min_child_samples"]
        ),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *space["colsample_bytree"]
        ),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
    }


def tune_hyperparameters(
    train: pd.DataFrame, params: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Busca los mejores hiperparámetros de Ridge y LightGBM con Optuna.

    Args:
        train: Conjunto de entrenamiento.
        params: Parámetros ``model_training``.

    Returns:
        Mejores hiperparámetros y MAE de validación por modelo, y la tabla con
        todos los trials (para documentar la búsqueda en el reporte).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    features = get_feature_columns(train, params)
    x, y = train[features], train[params["target"]]
    folds = _time_folds(train["Date"], params["cv_splits"])
    seed = params["random_state"]
    space = params["optuna"]

    def ridge_objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", *space["ridge"]["alpha"], log=True)
        return _cv_mae(lambda: build_ridge(alpha), x, y, folds)

    def lightgbm_objective(trial: optuna.Trial) -> float:
        hyper = _suggest_lightgbm(trial, space["lightgbm"])
        return _cv_mae(lambda: build_lightgbm(hyper, seed), x, y, folds)

    best: dict[str, Any] = {}
    trials = []
    for name, objective in [
        ("ridge", ridge_objective),
        ("lightgbm", lightgbm_objective),
    ]:
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=space["n_trials"])
        best[name] = {"params": study.best_params, "cv_mae": study.best_value}
        trials.append(study.trials_dataframe().assign(model=name))
        logger.info("%s: mejor MAE CV = %.0f", name, study.best_value)

    return best, pd.concat(trials, ignore_index=True)


def train_models(
    train: pd.DataFrame, best_params: dict[str, Any], params: dict[str, Any]
) -> tuple[Pipeline, lgb.LGBMRegressor]:
    """Entrena Ridge y LightGBM finales con todo el entrenamiento.

    Args:
        train: Conjunto de entrenamiento.
        best_params: Salida de ``tune_hyperparameters``.
        params: Parámetros ``model_training``.

    Returns:
        Modelo Ridge y modelo LightGBM entrenados.
    """
    features = get_feature_columns(train, params)
    x, y = train[features], train[params["target"]]
    ridge = build_ridge(best_params["ridge"]["params"]["alpha"]).fit(x, y)
    lightgbm = build_lightgbm(
        best_params["lightgbm"]["params"], params["random_state"]
    ).fit(x, y)
    return ridge, lightgbm
