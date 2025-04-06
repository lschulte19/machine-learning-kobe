import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def separar_variaveis(df: pd.DataFrame):
    X = df.drop(columns=["shot_made_flag"])
    y = df["shot_made_flag"]
    return X, y

def treinar_modelo(X: pd.DataFrame, y: pd.Series) -> bytes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return pickle.dumps(modelo)
