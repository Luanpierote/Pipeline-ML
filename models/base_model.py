"""Classe abstrata base para todos os modelos do pipeline."""
#  Contrato padrão que todo o modelo do pipeline precisa seguir
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
 
class BaseModel(ABC):
    """
    Interface comum a todos os modelos.
    Todo novo modelo deve herdar desta classe e implementar fit() e run().
    """
 
    def __init__(self, config: dict):
        self.config = config
        self.result_ = None
 
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Treina / ajusta o modelo ao DataFrame pré-processado."""
       
        coluna_alvo = self.config.get("target_column", df.columns[-1])

        # Valida se a coluna alvo existe no DataFrame
        if coluna_alvo not in df.columns:
            raise ValueError(f"Coluna alvo '{coluna_alvo}' não encontrada no DataFrame.")

        # Separa os dados de entrada (X) e os rótulos/valores alvo (y)        
        X = df.drop(columns=[coluna_alvo]) 
        y = df[coluna_alvo]

        max_depth = self.config.get("max_depth", 10)
        
        # Estimator com self. para ele ficar salvo no modelo
        self.estimator = RandomForestClassifier(max_depth=max_depth, random_state=42)
        self.estimator.fit(X, y)        
        
        # 3. Correção: self.result_ (com underline no final, igual estava na sua classe abstrata BaseModel)
        # E corrigido para X.columns maiúsculo
        self.result_ = f"RandomForest treinado com sucesso. Features utilizadas: {list(X.columns)}"

        return self
        ...
 
    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa o modelo e retorna o DataFrame com as predições/rótulos."""
        ...
 
    def summary(self) -> dict:
        """Retorna um resumo dos resultados. Pode ser sobrescrito por subclasses."""
        return {"model": self.__class__.__name__, "result": str(self.result_)}
 