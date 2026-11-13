"""Pruebas de integración del proyecto Kedro.

``test_full_pipeline_runs_on_sample`` ejecuta ``kedro run`` completo en el
entorno ``ci`` (muestra de ``data/00_sample``, resultados en ``data/ci``), con
pocos trials de Optuna para que sea rápido.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT = Path(__file__).resolve().parents[1]
CI_OUTPUT = PROJECT / "data" / "ci"
EXPECTED_PIPELINES = {
    "data_processing",
    "feature_engineering",
    "model_training",
    "model_evaluation",
    "sales_analysis",
    "business_impact",
}


@pytest.fixture(scope="module")
def ci_run():
    """Corre el pipeline completo una sola vez para todas las pruebas del módulo."""
    bootstrap_project(PROJECT)
    with KedroSession.create(
        project_path=PROJECT,
        env="ci",
        runtime_params={"model_training": {"optuna": {"n_trials": 2}}},
    ) as session:
        session.run()
    return CI_OUTPUT


def test_pipelines_are_registered():
    """Todos los pipelines se descubren y el default los incluye."""
    bootstrap_project(PROJECT)
    assert EXPECTED_PIPELINES <= set(pipelines)
    default_nodes = {n.name for n in pipelines["__default__"].nodes}
    for name in EXPECTED_PIPELINES:
        assert {n.name for n in pipelines[name].nodes} <= default_nodes


def test_full_pipeline_runs_on_sample(ci_run):
    """El pipeline completo produce los entregables principales."""
    reporting = ci_run / "08_reporting"
    for name in [
        "data_quality_report.json",
        "model_metrics.csv",
        "model_selection.json",
        "seller_summary.csv",
        "sales_insights.json",
        "shap_summary.png",
        "seller_contribution.png",
        "business_impact.json",
    ]:
        assert (reporting / name).exists(), name
    assert (ci_run / "06_models" / "lightgbm_model.pkl").exists()


def test_sample_outputs_are_consistent(ci_run):
    """Las salidas de la muestra tienen sentido de negocio."""
    quality = json.loads(
        (ci_run / "08_reporting" / "data_quality_report.json").read_text()
    )
    assert quality["rows_clean"] <= quality["rows_raw"]
    assert quality["total_revenue"] > 0

    summary = pd.read_csv(
        ci_run / "08_reporting" / "seller_summary.csv", encoding="latin-1"
    )
    assert summary["SharePct"].sum() == pytest.approx(100.0)

    selection = json.loads(
        (ci_run / "08_reporting" / "model_selection.json").read_text()
    )
    metrics = pd.read_csv(
        ci_run / "08_reporting" / "model_metrics.csv", encoding="latin-1"
    )
    assert selection["selected_model"] in set(metrics["model"])
