from kedro.pipeline import Pipeline, node, pipeline
from .nodes import treinar_modelos


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=treinar_modelos,
            inputs="base_train",
            outputs="modelo_treinado",
            name="treinar_modelos_node"
        )
    ])
