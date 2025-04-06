import mlflow
from pycaret.classification import setup, compare_models, pull, save_model
import pandas as pd


def treinar_modelos(base_train: pd.DataFrame):
    mlflow.set_experiment("Treinamento")
    with mlflow.start_run(run_name="TreinamentoModelos"):

        # Setup PyCaret
        s = setup(
            data=base_train,
            target='shot_made_flag',
            session_id=123,
            log_experiment=False,
            use_gpu=False,
            verbose=False,
        )

        # Comparar modelos
        best_model = compare_models(include=["lr", "dt"])
        leaderboard = pull()

        # Registrar leaderboard como artefato
        leaderboard.to_csv("leaderboard.csv", index=False)
        mlflow.log_artifact("leaderboard.csv")

        # Registrar métricas
        best_model_name = leaderboard.iloc[0]["Model"]
        mlflow.log_param("melhor_modelo", best_model_name)

        # Salvar modelo
        save_model(best_model, "modelo_final")
        mlflow.log_artifact("modelo_final.pkl")

        return best_model
