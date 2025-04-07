from kedro.pipeline import Pipeline, node
from .nodes import preparar_dados

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preparar_dados,
            inputs="dataset_kobe_prod",
            outputs="dataset_producao",
            name="preparar_dados_producao_node"
        ),
    ])