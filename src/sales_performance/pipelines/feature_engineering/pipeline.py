"""Definición del pipeline ``feature_engineering``."""

from kedro.pipeline import Node, Pipeline

from .nodes import build_features, build_seller_daily, split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline que va de ``sales_primary`` a los conjuntos de modelado.

    Returns:
        Pipeline de agregación vendedor-día, features y split temporal.
    """
    return Pipeline(
        [
            Node(
                func=build_seller_daily,
                inputs="sales_primary",
                outputs="seller_daily",
                name="build_seller_daily",
            ),
            Node(
                func=build_features,
                inputs=[
                    "seller_daily",
                    "employees_clean",
                    "params:feature_engineering",
                ],
                outputs="seller_features",
                name="build_features",
            ),
            Node(
                func=split_train_test,
                inputs=["seller_features", "params:feature_engineering"],
                outputs=["model_input_train", "model_input_test"],
                name="split_train_test",
            ),
        ]
    )
