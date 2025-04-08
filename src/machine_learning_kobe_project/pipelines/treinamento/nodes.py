import mlflow
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, predict_model
from sklearn.metrics import log_loss, f1_score


def treinar_modelos(base_train: pd.DataFrame, base_test: pd.DataFrame):
    mlflow.set_experiment("Treinamento")
    with mlflow.start_run(run_name="TreinamentoModelos"):

        # Remover alvos nulos
        base_train = base_train.dropna(subset=["shot_made_flag"])
        base_test = base_test.dropna(subset=["shot_made_flag"])

        # PyCaret setup
        setup(
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

        leaderboard.to_csv("leaderboard.csv", index=False)
        mlflow.log_artifact("leaderboard.csv")

        best_model_name = leaderboard.iloc[0]["Model"]
        mlflow.log_param("melhor_modelo", best_model_name)

        pred = predict_model(best_model, data=base_test)
        y_true = pred['shot_made_flag']
        y_pred = pred['prediction_label']
        y_prob = pred['prediction_score']

        mlflow.log_metric("log_loss", log_loss(y_true, y_prob))
        mlflow.log_metric("f1_score", f1_score(y_true, y_pred))

        # Salvar modelo final
        save_model(best_model, "modelo_final")
        mlflow.log_artifact("modelo_final.pkl")

        return best_model
