import pandas as pd

def preparar_dados(base_producao: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o pré-processamento da base de produção da mesma forma que foi feito na base de treino.

    Parâmetros:
        base_producao (pd.DataFrame): DataFrame contendo os dados brutos de produção.

    Retorno:
        pd.DataFrame: DataFrame pré-processado.
    """

  

    df = base_producao.copy()

    colunas_para_remover = ["game_id", "team_id", "team_name"]
    df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns], errors="ignore")

    df = df.fillna(0)

    return df
