"""Definición del pipeline ``sales_analysis``."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_seller_summary,
    compute_insights,
    plot_discount_effect,
    plot_profile_vs_performance,
    plot_seller_contribution,
    plot_seller_trend,
    plot_volume_vs_ticket,
    plot_weekday_pattern,
    plot_weekly_trend,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de análisis descriptivo de ventas y vendedores.

    Returns:
        Pipeline que produce la ficha por vendedor, las cifras clave y gráficas.
    """
    return Pipeline(
        [
            Node(
                func=build_seller_summary,
                inputs=["seller_daily", "employees_clean", "params:sales_analysis"],
                outputs="seller_summary",
                name="build_seller_summary",
            ),
            Node(
                func=compute_insights,
                inputs=[
                    "seller_daily",
                    "sales_primary",
                    "seller_summary",
                    "params:sales_analysis",
                ],
                outputs="sales_insights",
                name="compute_insights",
            ),
            Node(plot_weekly_trend, "seller_daily", "plot_weekly_trend"),
            Node(plot_weekday_pattern, "seller_daily", "plot_weekday_pattern"),
            Node(
                plot_seller_contribution, "seller_summary", "plot_seller_contribution"
            ),
            Node(plot_seller_trend, "seller_summary", "plot_seller_trend"),
            Node(plot_volume_vs_ticket, "seller_summary", "plot_volume_vs_ticket"),
            Node(plot_discount_effect, "sales_primary", "plot_discount_effect"),
            Node(
                plot_profile_vs_performance,
                ["seller_summary", "params:sales_analysis"],
                "plot_profile_vs_performance",
            ),
        ]
    )
