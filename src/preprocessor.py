""" LIMPEZA DO DATASET, PADRONIZAÇÃO E REDIMENSIONAMENTO """

# EM DESENVOLVIMENTO ⌛
# PROXIMA ETAPA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

# @staticmethod - é um decorator do python que indica que o método não depende da instância(self) nem da classe para funcionar. pode chamar direto pela classe também

class Preprocessor:

    df = pd.DataFrame({
        "nome": ["Ana", "Carlos"],
        "idade": [20, 30]
    })

#Remoção de colunas desnecessárias 
    @staticmethod
    def drop_col(
        df                : pd.DataFrame,
        coluna_alvo       : str,
        limite_nulos      : float = 0.5,
        limite_variancia  : float = 0.0,
        limite_correlacao : float = 0.95,
        verbose           : bool  = True
    ) -> pd.DataFrame:
        """Remove colunas irrelevantes por nulos, variância baixa e alta correlação."""
        df_copia = df.copy()

        proporcao_nulos = df_copia.isnull().mean()
        colunas_nulos   = proporcao_nulos[proporcao_nulos > limite_nulos].index.tolist()

        df_copia.drop(columns=colunas_nulos, inplace=True)

        if verbose and colunas_nulos:
            print(f"[nulos]      Removidas (> {limite_nulos*100:.0f}% nulos): {colunas_nulos}")

        cols_numericas      = df_copia.select_dtypes(include=np.number).columns.tolist()
        cols_para_variancia = [c for c in cols_numericas if c != coluna_alvo]

        if cols_para_variancia:
            seletor           = VarianceThreshold(threshold=limite_variancia)
            seletor.fit(df_copia[cols_para_variancia])
            colunas_variancia = [
                col for col, mantida in zip(cols_para_variancia, seletor.get_support())
                if not mantida
            ]

            df_copia.drop(columns=colunas_variancia, inplace=True)

            if verbose and colunas_variancia:
                print(f"[variância]  Removidas (< {limite_variancia}): {colunas_variancia}")

        cols_para_corr = [
            c for c in df_copia.columns
            if c != coluna_alvo and df_copia[c].dtype != object
        ]

        if not cols_para_corr:
            if verbose:
                print("[correlação] Nenhuma feature numérica restante para verificar.")
            return df_copia

        matriz_corr   = df_copia[cols_para_corr].corr().abs()
        triangulo_sup = matriz_corr.where(
            np.triu(np.ones(matriz_corr.shape), k=1).astype(bool)
        )
        colunas_corr = [
            col for col in triangulo_sup.columns
            if any(triangulo_sup[col] > limite_correlacao)
        ]

        df_copia.drop(columns=colunas_corr, inplace=True)

        if verbose and colunas_corr:
            print(f"[correlação] Removidas (> {limite_correlacao}): {colunas_corr}")

        if verbose:
            total_removidas = len(colunas_nulos) + len(colunas_variancia) + len(colunas_corr)
            print(f"\nTotal removidas: {total_removidas} | Colunas restantes: {df_copia.shape[1]}")
        return df_copia
        
# Lógica para imputar valores nulos usando diferentes estratégias
    @staticmethod
    def impute(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        if strategy == 'mean':
            return df.fillna(df.mean(numeric_only=True)) 
        elif strategy == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif strategy == 'mode':
            for col in df.columns:
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
            return df 
        elif strategy == 'zero':
            return df.fillna(0)
        else:
            raise ValueError(f"Estratégia '{strategy}' não reconhecida. Use 'mean', 'median', 'mode' ou 'zero'.") 
             
# encode — codificação de variáveis categóricas (label, onehot)
    @staticmethod
    def encode(df,method):

        for coluna in df.columns:
            if df[coluna].dtype == "object":
                if method == "label":
                    # Transforma cada categoria em um valor inteiro
                    label = LabelEncoder()
                    df[coluna] = label.fit_transform(df[coluna])
                elif method == "onehot":
                    # Cria uma nova coluna para cada categoria
                    onehot = OneHotEncoder(handle_unknown='ignore')
                    encoded = onehot.fit_transform(df[[coluna]])
                    encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out([coluna]))
                    df = df.drop(columns = [coluna]).join(encoded_df)
        return df

# print(Preprocessor.encode(df,encoder))           


    
        
# scale — escalonamento de variáveis numéricas (standard, minmax)
    # def scale():
    @staticmethod
    def scale(df: pd.DataFrame, scale_map: dict):

        df = df.copy()

        for coluna, scaler_class in scale_map.items():

            # verifica se a coluna existe
            if coluna not in df.columns:
                print(f"Coluna '{coluna}' não encontrada")
                continue

            # cria instância do scaler
            scaler = scaler_class()

            # aplica transformação
            df[coluna] = scaler.fit_transform(df[[coluna]])

        return df
    
class DataPreprocessor:

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        # 1. Remove colunas desnecessárias
        coluna_alvo = self.config["required"][-1]
        df = Preprocessor.drop_col(df, coluna_alvo=coluna_alvo)

        # 2. Imputa nulos nas colunas críticas
        df = Preprocessor.impute(None, df, strategy="mean")

        # 3. Codifica colunas categóricas conforme o config
        encode_map = self.config.get("preprocessing", {}).get("encode", {})
        for coluna, metodo in encode_map.items():
            if coluna in df.columns:
                df = Preprocessor.encode(df, method=metodo)

        return df


preprocessor = DataPreprocessor("../config.json")
df_limpo = preprocessor.run(df)


