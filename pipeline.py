import pandas as pd
import mlflow
import os

# Ativa o tracking
mlflow.set_experiment("Kobe_Model")
with mlflow.start_run(run_name="PreparacaoDados"):
    # Lê os dois datasets
    dev_df = pd.read_parquet("data/raw/dataset_kobe_dev.parquet")
    prod_df = pd.read_parquet("data/raw/dataset_kobe_prod.parquet")
    
    # Concatena
    df = pd.concat([dev_df, prod_df])

    # Seleciona colunas
    cols = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    df_filtered = df[cols]

    # Remove linhas com dados faltantes
    df_filtered = df_filtered.dropna()

    # Salva resultado
    output_path = "data/processed/data_filtered.parquet"
    os.makedirs("data/processed", exist_ok=True)
    df_filtered.to_parquet(output_path, index=False)

    # Log do artefato e dimensão
    mlflow.log_artifact(output_path)
    mlflow.log_param("num_rows", df_filtered.shape[0])
    mlflow.log_param("num_cols", df_filtered.shape[1])
