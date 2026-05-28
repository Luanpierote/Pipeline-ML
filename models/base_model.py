"""Classe abstrata base para todos os modelos do pipeline."""
#  Contrato padrão que todo o modelo do pipeline precisa seguir
from abc import ABC, abstractmethod
import pandas as pd
 
 
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
        ...
 
    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa o modelo e retorna o DataFrame com as predições/rótulos."""
        ...
 
    def summary(self) -> dict:
        """Retorna um resumo dos resultados. Pode ser sobrescrito por subclasses."""
        return {"model": self.__class__.__name__, "result": str(self.result_)}
 