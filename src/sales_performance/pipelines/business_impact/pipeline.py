"""Definición del pipeline ``business_impact``."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    estimate_discount_impact,
    estimate_goal_accuracy,
    estimate_upsell_impact,
    summarize_impact,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de estimación del impacto de negocio.

    Returns:
        Pipeline que produce ``business_impact.json``, la tabla de venta
        sugestiva por vendedor y la gráfica de impacto anual.
    """
    return Pipeline(
        [
            Node(
                func=estimate_discount_impact,
                inputs=["sales_primary", "params:business_impact"],
                outputs="impact_discounts",
                name="estimate_discount_impact",
            ),
            Node(
                func=estimate_upsell_impact,
                inputs=["sales_primary", "params:business_impact"],
                outputs=["impact_upselling", "upsell_by_seller"],
                name="estimate_upsell_impact",
            ),
            Node(
                func=estimate_goal_accuracy,
                inputs=["model_metrics", "model_selection", "params:business_impact"],
                outputs="impact_goals",
                name="estimate_goal_accuracy",
            ),
            Node(
                func=summarize_impact,
                inputs=["impact_discounts", "impact_upselling", "impact_goals"],
                outputs=["business_impact", "plot_business_impact"],
                name="summarize_impact",
            ),
        ]
    )
