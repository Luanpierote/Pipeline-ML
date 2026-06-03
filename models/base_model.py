"""Classe abstrata base para todos os modelos do pipeline."""
#  Contrato padrão que todo o modelo do pipeline precisa seguir
# Princípio: ela não sabe qual algoritmo irá usar
from abc import ABC, abstractmethod
import importlib
import pandas as pd
 
 
class BaseModel(ABC):
    """
    Interface comum a todos os modelos.
    Todo novo modelo deve herdar desta classe e implementar fit() e run().
    """
 
    def __init__(self, config: dict):
        self.config = config
        self.result_ = None
        self._model = self._resolve_model()

    def _resolve_model(self):
        """
        Instancia qualquer classe sklearn a partir das intruções do config(o importlib possibilita isso).
        Espera no config:
            "model_class"  : "sklearn.cluster.DBSCAN"
            "model_params" : { "eps": 0.5, "min_samples": 5 }
        Se não houver model_class, retorna None (subclasse cuida disso).
        """
        class_path = self.config.get("model_class")
        if not class_path:
            return None

        # Importação automática
        try:
            module_path , class_name = class_path.rsplit(".",1)
            cls = getattr(importlib.import_module(module_path),class_name)
            params = self.config.get("model_params", {})
            return cls(**params)
        
        except (ImportError,AttributeError) as e:
            raise ImportError(f"[BaseModel] Não foi possível carregar '{class_path}': {e}")
    
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


# fit()            → cada subclasse implementa como quiser (X/y, fit_predict, pipeline com scaler...)
# run()            → retorna o DataFrame do jeito que o modelo precisa
# summary()        → sobrescreve com as métricas específicas (silhouette, F1, AUC...)
# DataPreprocessor → cuida do scaler antes de chegar no modelo