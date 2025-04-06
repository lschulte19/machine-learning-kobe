from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filtrar_variaveis, separar_treino_teste

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=filtrar_variaveis,
            inputs="dataset_kobe_dev",
            outputs="data_filtered",
            name="filtrar_variaveis_node",
        ),
        node(
            func=separar_treino_teste,
            inputs=dict(df="data_filtered", test_size="params:test_size"),
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="separar_treino_teste_node",
        ),
    ])
