"""Pruebas de integración del proyecto Kedro."""

from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_pipelines_are_registered():
    """Los pipelines del proyecto se descubren y el default no está vacío."""
    bootstrap_project(Path.cwd())
    assert "data_processing" in pipelines
    assert len(pipelines["__default__"].nodes) > 0
