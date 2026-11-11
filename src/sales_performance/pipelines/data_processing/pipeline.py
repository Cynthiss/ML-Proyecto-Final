"""Definición del pipeline ``data_processing``."""

from kedro.pipeline import Node, Pipeline

from .nodes import build_sales_primary, clean_employees, clean_products, clean_sales


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de limpieza y unión de tablas.

    Returns:
        Pipeline que va de las tablas crudas a ``sales_primary``.
    """
    return Pipeline(
        [
            Node(
                func=clean_sales,
                inputs=["raw_sales", "params:data_processing"],
                outputs=["sales_clean", "sales_quality_partial"],
                name="clean_sales",
            ),
            Node(
                func=clean_products,
                inputs=["raw_products", "raw_categories"],
                outputs="products_clean",
                name="clean_products",
            ),
            Node(
                func=clean_employees,
                inputs=["raw_employees", "raw_cities", "params:data_processing"],
                outputs="employees_clean",
                name="clean_employees",
            ),
            Node(
                func=build_sales_primary,
                inputs=[
                    "sales_clean",
                    "products_clean",
                    "employees_clean",
                    "raw_customers",
                    "sales_quality_partial",
                ],
                outputs=["sales_primary", "data_quality_report"],
                name="build_sales_primary",
            ),
        ]
    )
