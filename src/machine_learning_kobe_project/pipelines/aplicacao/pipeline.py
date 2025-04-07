from kedro.pipeline import Pipeline, node, pipeline
from .nodes import aplicar_modelo


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=aplicar_modelo,
            inputs="dataset_producao",          
            outputs="resultado_aplicacao",     
            name="aplicar_modelo_node"
        )
    ])
