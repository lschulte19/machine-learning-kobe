import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    # Remove colunas irrelevantes para o modelo
    colunas_para_remover = ["team_id", "game_event_id", "game_id", "team_name"]
    df = df.drop(columns=colunas_para_remover, errors="ignore")

    # Remove linhas com valores ausentes
    df = df.dropna()

    return df

def separar_treino_teste(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["shot_made_flag"],
        random_state=42
    )

    # Log no MLflow
    mlflow.set_experiment("Preprocessamento")
    with mlflow.start_run(run_name="SplitTreinoTeste"):
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("tamanho_treino", len(X_train))
        mlflow.log_metric("tamanho_teste", len(X_test))

    return X_train, X_test
