""" LIMPEZA DO DATASET, PADRONIZAÇÃO E REDIMENSIONAMENTO """

# EM DESENVOLVIMENTO ⌛
# PROXIMA ETAPA
import pandas as pd
class Preprocessor:

# drop_cols — remoção de colunas desnecessárias (IDs, irrelevantes)
    # def drop_cols():
        
# Lógica para imputar valores nulos usando diferentes estratégias
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
    #  def encode():

# scale — escalonamento de variáveis numéricas (standard, minmax)
    # def scale():
        
# class Datapreprocessor:
    # ler o config gerado pelo agente, garantir a ordem correta das etapas
    # e devolver um DataFrame limpo e pronto para o modelo.
