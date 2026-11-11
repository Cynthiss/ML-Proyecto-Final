"""Pruebas de los nodos de ``model_evaluation``."""

import pandas as pd
import pytest

from sales_performance.pipelines.model_evaluation.nodes import evaluate_models

PARAMS = {
    "target": "Revenue",
    "min_improvement": 0.02,
    "baselines": {"Ayer": "Revenue_lag1", "Promedio": "Revenue_expanding_mean"},
}
BEST = {"ridge": {"cv_mae": 1.0}, "lightgbm": {"cv_mae": 1.0}}


def _predictions(ridge_error: float) -> pd.DataFrame:
    real = pd.Series([100.0, 100.0, 100.0, 100.0])
    return pd.DataFrame(
        {
            "Revenue": real,
            "Ayer": real + 20,  # MAE 20
            "Promedio": real + 10,  # MAE 10 -> mejor línea base
            "Ridge": real + ridge_error,
            "LightGBM": real + 15,
        }
    )


def test_ml_model_selected_when_clearly_better():
    metrics, selection = evaluate_models(_predictions(5.0), BEST, PARAMS)
    assert selection["selected_model"] == "Ridge"
    assert selection["best_baseline"] == "Promedio"
    assert selection["ml_improvement_%"] == pytest.approx(50.0)
    assert metrics.iloc[0]["model"] == "Ridge"


def test_baseline_selected_when_improvement_is_marginal():
    # Ridge mejora solo 1% (MAE 9.9 vs 10): no alcanza el umbral de 2%
    _, selection = evaluate_models(_predictions(9.9), BEST, PARAMS)
    assert selection["selected_model"] == "Promedio"
    assert "más simple" in selection["reason"]
