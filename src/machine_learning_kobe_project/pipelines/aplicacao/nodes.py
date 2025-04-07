import pandas as pd
import mlflow
from pycaret.classification import load_model, predict_model
from sklearn.metrics import log_loss, f1_score
import datetime
import numpy as np


def aplicar_modelo(base_producao: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_experiment("Aplicacao")

    with mlflow.start_run(run_name="PipelineAplicacao"):
        # Define timestamp para salvar artefato com nome único
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_modelo = "modelo_final"

        # Log de parâmetros iniciais
        mlflow.set_tag("pipeline", "aplicacao")
        mlflow.set_tag("modelo", nome_modelo)
        mlflow.log_param("timestamp_execucao", timestamp)

        # Carrega o modelo treinado
        modelo = load_model(nome_modelo)

        # Aplica predições
        resultado = predict_model(modelo, data=base_producao)

        # Avalia métricas caso a base contenha rótulo
        if 'shot_made_flag' in resultado.columns:
            y_true = resultado['shot_made_flag']
            y_pred = resultado['prediction_label']
            y_prob = resultado['prediction_score']

            mlflow.log_metric("log_loss_aplicacao", log_loss(y_true, y_prob))
            mlflow.log_metric("f1_score_aplicacao", f1_score(y_true, y_pred))

        # Log de distribuição das predições
        mlflow.log_metric("predicoes_classe_0", int((resultado['prediction_label'] == 0).sum()))
        mlflow.log_metric("predicoes_classe_1", int((resultado['prediction_label'] == 1).sum()))
        mlflow.log_metric("percentual_classe_1", float(np.mean(resultado['prediction_label'] == 1)))

        # Salva e loga artefato
        filename = f"resultado_aplicacao_{timestamp}.parquet"
        resultado.to_parquet(filename, index=False)
        mlflow.log_artifact(filename)

        return resultado
