"""Nodos del pipeline ``model_evaluation``.

Compara los modelos contra líneas base simples en el conjunto de prueba
(últimos días, nunca vistos), elige el modelo con un criterio de negocio
y explica sus predicciones con SHAP.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from sales_performance.pipelines.model_training.nodes import (  # noqa: E402
    get_feature_columns,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = {"ridge": "Ridge", "lightgbm": "LightGBM"}


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcula MAE, RMSE, MAPE y R² de un pronóstico."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def predict_test(
    test: pd.DataFrame,
    ridge_model,
    lightgbm_model,
    training_params: dict[str, Any],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Genera los pronósticos de modelos y líneas base sobre el conjunto de prueba.

    Args:
        test: Conjunto de prueba.
        ridge_model: Modelo Ridge entrenado.
        lightgbm_model: Modelo LightGBM entrenado.
        training_params: Parámetros ``model_training`` (columnas del modelo).
        params: Parámetros ``model_evaluation`` (líneas base).

    Returns:
        Tabla vendedor-día con el valor real y una columna por pronóstico.
    """
    features = get_feature_columns(test, training_params)
    preds = test[["SalesPersonID", "Date", params["target"]]].copy()
    for name, column in params["baselines"].items():
        preds[name] = test[column]
    preds[MODEL_NAMES["ridge"]] = ridge_model.predict(test[features])
    preds[MODEL_NAMES["lightgbm"]] = lightgbm_model.predict(test[features])
    return preds


def evaluate_models(
    predictions: pd.DataFrame, best_params: dict[str, Any], params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calcula métricas de cada pronóstico y elige el modelo final.

    Regla de selección: un modelo de ML solo se elige si reduce el MAE de la
    mejor línea base en al menos ``min_improvement``. Si no, se recomienda la
    línea base (más simple de explicar, implementar y mantener).

    Args:
        predictions: Salida de ``predict_test``.
        best_params: Hiperparámetros y MAE de validación de Optuna.
        params: Parámetros ``model_evaluation``.

    Returns:
        Tabla de métricas (ordenada por MAE) y el resumen de la selección.
    """
    target = params["target"]
    baselines = list(params["baselines"])
    rows = []
    for name in [*baselines, *MODEL_NAMES.values()]:
        rows.append(
            {
                "model": name,
                "type": "Línea base" if name in baselines else "Modelo ML",
                **_metrics(predictions[target], predictions[name]),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    best_baseline = metrics[metrics["type"] == "Línea base"].iloc[0]
    best_ml = metrics[metrics["type"] == "Modelo ML"].iloc[0]
    improvement = 1 - best_ml["MAE"] / best_baseline["MAE"]
    use_ml = improvement >= params["min_improvement"]
    selected = best_ml["model"] if use_ml else best_baseline["model"]

    metrics["improvement_vs_best_baseline_%"] = 100 * (
        1 - metrics["MAE"] / best_baseline["MAE"]
    )
    selection = {
        "selected_model": selected,
        "best_baseline": best_baseline["model"],
        "best_baseline_mae": round(float(best_baseline["MAE"]), 2),
        "best_ml_model": best_ml["model"],
        "best_ml_mae": round(float(best_ml["MAE"]), 2),
        "ml_improvement_%": round(100 * float(improvement), 2),
        "min_improvement_%": 100 * params["min_improvement"],
        "mean_daily_revenue_test": round(float(predictions[target].mean()), 2),
        "cv_mae_optuna": {
            MODEL_NAMES[k]: round(v["cv_mae"], 2) for k, v in best_params.items()
        },
        "reason": (
            f"{best_ml['model']} tiene un MAE {abs(100 * improvement):.2f}% "
            f"{'menor' if improvement > 0 else 'mayor'} que "
            f"'{best_baseline['model']}'"
            + (
                "; supera el umbral y se elige el modelo de ML."
                if use_ml
                else f"; no alcanza el umbral de {100 * params['min_improvement']:.0f}%, "
                "por lo que se recomienda la alternativa más simple."
            )
        ),
    }
    logger.info("Selección: %s", selection["reason"])
    return metrics, selection


def explain_model(
    lightgbm_model,
    test: pd.DataFrame,
    training_params: dict[str, Any],
    feature_params: dict[str, Any],
) -> tuple[pd.DataFrame, plt.Figure]:
    """Explica el modelo LightGBM con valores SHAP.

    Muestra qué variables mueven el pronóstico y, en particular, cuánto pesa el
    perfil del vendedor (edad y antigüedad) frente a su historial.

    Args:
        lightgbm_model: Modelo LightGBM entrenado.
        test: Conjunto de prueba.
        training_params: Parámetros ``model_training``.
        feature_params: Parámetros ``feature_engineering`` (``profile_features``).

    Returns:
        Importancia media |SHAP| por variable y la gráfica resumen.
    """
    features = get_feature_columns(test, training_params)
    x = test[features]
    shap_values = shap.TreeExplainer(lightgbm_model).shap_values(x)

    importance = (
        pd.DataFrame(
            {"feature": features, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    total = importance["mean_abs_shap"].sum()
    importance["share_%"] = 100 * importance["mean_abs_shap"] / total if total else 0.0
    importance["is_profile"] = importance["feature"].isin(
        feature_params["profile_features"]
    )

    plt.figure()
    shap.summary_plot(shap_values, x, show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle("¿Qué variables mueven el pronóstico? (SHAP)")
    return importance, fig


def plot_forecast_vs_actual(
    predictions: pd.DataFrame, selection: dict[str, Any], params: dict[str, Any]
) -> plt.Figure:
    """Grafica la venta total diaria real vs. pronosticada en el periodo de prueba."""
    target = params["target"]
    cols = [target, selection["best_baseline"], selection["best_ml_model"]]
    daily = predictions.groupby("Date")[cols].sum() / 1e6

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily[target], color="black", lw=2, label="Venta real")
    ax.plot(daily.index, daily[cols[1]], ls="--", label=f"Línea base: {cols[1]}")
    ax.plot(daily.index, daily[cols[2]], ls=":", lw=2, label=f"Modelo: {cols[2]}")
    ax.set_title("Venta diaria total: real vs. pronóstico (periodo de prueba)")
    ax.set_ylabel("Millones")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Grafica el MAE de cada modelo y línea base (menor es mejor)."""
    data = metrics.sort_values("MAE", ascending=False)
    colors = ["#4C72B0" if t == "Modelo ML" else "#BBBBBB" for t in data["type"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["model"], data["MAE"] / 1e3, color=colors)
    for y, v in enumerate(data["MAE"] / 1e3):
        ax.text(v, y, f" {v:,.1f}", va="center")
    ax.set_xlabel("MAE (miles) — error promedio diario por vendedor")
    ax.set_title("Comparación de modelos vs. líneas base (gris)")
    fig.tight_layout()
    return fig
