"""Definición del pipeline ``model_training``."""

from kedro.pipeline import Node, Pipeline

from .nodes import train_models, tune_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de optimización con Optuna y entrenamiento final.

    Returns:
        Pipeline que produce los modelos Ridge y LightGBM entrenados.
    """
    return Pipeline(
        [
            Node(
                func=tune_hyperparameters,
                inputs=["model_input_train", "params:model_training"],
                outputs=["best_params", "optuna_trials"],
                name="tune_hyperparameters",
            ),
            Node(
                func=train_models,
                inputs=["model_input_train", "best_params", "params:model_training"],
                outputs=["ridge_model", "lightgbm_model"],
                name="train_models",
            ),
        ]
    )
