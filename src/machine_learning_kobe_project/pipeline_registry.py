from kedro.pipeline import Pipeline
from machine_learning_kobe_project.pipelines.preprocesso import create_pipeline as preprocesso_pipeline
from machine_learning_kobe_project.pipelines.treinamento import create_pipeline as treinamento_pipeline


...

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "preprocesso": preprocesso_pipeline(),
        "treinamento": treinamento_pipeline(),
        "__default__": preprocesso_pipeline() + treinamento_pipeline(),
    }
