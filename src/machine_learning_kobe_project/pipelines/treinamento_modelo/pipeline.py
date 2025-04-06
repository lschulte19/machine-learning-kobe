from kedro.pipeline import Pipeline, node, pipeline
from .nodes import separar_variaveis, treinar_modelo

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=separar_variaveis,
                inputs="data_filtered",
                outputs=["X", "y"],
                name="separar_variaveis_node"
            ),
            node(
                func=treinar_modelo,
                inputs=["X", "y"],
                outputs="modelo_final",
                name="treinar_modelo_node"
            )
        ]
    )
