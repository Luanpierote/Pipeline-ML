""" LIMPEZA DO DATASET, PADRONIZAÇÃO E REDIMENSIONAMENTO """

# EM DESENVOLVIMENTO ⌛
# PROXIMA ETAPA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import pandas as pd

# @staticmethod - é um decorator do python que indica que o método não depende da instância(self) nem da classe para funcionar. pode chamar direto pela classe também

class Preprocessor:

    df = pd.DataFrame({
        "nome": ["Ana", "Carlos"],  
        "idade": [20, 30]
    })

# drop_cols — remoção de colunas desnecessárias (IDs, irrelevantes)
    # def drop_cols():
        
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
        
# class Datapreprocessor:
    # ler o config gerado pelo agente, garantir a ordem correta das etapas
    # e devolver um DataFrame limpo e pronto para o modelo.
