import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def filtrar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["shot_made_flag"])

def separar_treino_teste(df: pd.DataFrame, test_size: float, random_state: int = 42):
    y = df["shot_made_flag"]
    X = df.drop(columns=["shot_made_flag"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Log para MLflow
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))

    return X_train, X_test, y_train, y_test
