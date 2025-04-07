from kedro.pipeline import Pipeline
from machine_learning_kobe_project.pipelines.preprocesso import create_pipeline as preprocesso_pipeline
from machine_learning_kobe_project.pipelines.treinamento import create_pipeline as treinamento_pipeline
from machine_learning_kobe_project.pipelines.aplicacao import create_pipeline as aplicacao_pipeline
from machine_learning_kobe_project.pipelines import preprocesso_producao
...

def register_pipelines():
    return {
        "preprocesso": preprocesso_pipeline(),
        "treinamento": treinamento_pipeline(),
        "aplicacao": aplicacao_pipeline(),
        "preprocesso_producao": preprocesso_producao.create_pipeline(),
        "__default__": preprocesso_pipeline() + treinamento_pipeline() + aplicacao_pipeline(),
    }