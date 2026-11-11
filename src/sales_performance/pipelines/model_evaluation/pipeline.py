"""Definición del pipeline ``model_evaluation``."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    evaluate_models,
    explain_model,
    plot_forecast_vs_actual,
    plot_model_comparison,
    predict_test,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de evaluación, selección e interpretación.

    Returns:
        Pipeline que produce métricas, selección del modelo, SHAP y gráficas.
    """
    return Pipeline(
        [
            Node(
                func=predict_test,
                inputs=[
                    "model_input_test",
                    "ridge_model",
                    "lightgbm_model",
                    "params:model_training",
                    "params:model_evaluation",
                ],
                outputs="test_predictions",
                name="predict_test",
            ),
            Node(
                func=evaluate_models,
                inputs=["test_predictions", "best_params", "params:model_evaluation"],
                outputs=["model_metrics", "model_selection"],
                name="evaluate_models",
            ),
            Node(
                func=explain_model,
                inputs=[
                    "lightgbm_model",
                    "model_input_test",
                    "params:model_training",
                    "params:feature_engineering",
                ],
                outputs=["feature_importance", "plot_shap_summary"],
                name="explain_model",
            ),
            Node(
                func=plot_forecast_vs_actual,
                inputs=[
                    "test_predictions",
                    "model_selection",
                    "params:model_evaluation",
                ],
                outputs="plot_forecast_vs_actual",
                name="plot_forecast_vs_actual",
            ),
            Node(
                func=plot_model_comparison,
                inputs="model_metrics",
                outputs="plot_model_comparison",
                name="plot_model_comparison",
            ),
        ]
    )
