"""Classe abstrata base para todos os modelos do pipeline."""
#  Contrato padrão que todo o modelo do pipeline precisa seguir
# Princípio: o Pipeline não sabe qual algoritmo irá usar, ele só vai puxar o método run
from abc import ABC, abstractmethod
# arquitetura baseada em configuração dinâmica 
import importlib
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
        self._model = self._resolve_model()

    def _resolve_model(self):
        """
        Instancia qualquer classe sklearn a partir das intruções do config(o importlib possibilita isso).
        Espera no config:
            "model_class"  : "sklearn.cluster.DBSCAN"
            "model_params" : { "eps": 0.5, "min_samples": 5 }
        Se não houver model_class, retorna None (subclasse cuida disso).
        """
        model_cfg = self.config.get("model", {})
        class_path = model_cfg.get("model_class")
        params = model_cfg.get("model_params", {})
        if not class_path:
            return None

        # Importação automática
        try:
            module_path , class_name = class_path.rsplit(".",1)
            cls = getattr(importlib.import_module(module_path),class_name)
            return cls(**params)
        
        except (ImportError,AttributeError) as e:
            raise ImportError(f"[BaseModel] Não foi possível carregar '{class_path}': {e}")
    
    def fit(self, df: pd.DataFrame):
        """Treina / ajusta o modelo ao DataFrame pré-processado."""
       
        coluna_alvo = self.config.get(
            "target_column",
            df.columns[-1]
        )

        if coluna_alvo not in df.columns:
            raise ValueError(
                f"Coluna alvo '{coluna_alvo}' não encontrada."
            )

        X = df.drop(columns=[coluna_alvo])
        y = df[coluna_alvo]

        self._model.fit(X, y)

        self.result_ = (
            f"Modelo treinado com sucesso. "
            f"Features utilizadas: {list(X.columns)}"
        )

        return self
        
 
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa o modelo e retorna o DataFrame com as predições/rótulos."""
        df_resultado = df.copy()
    
        if self._model is None:
            raise ValueError(
                "[BaseModel] Nenhum modelo foi carregado. "
                "Defina 'model_class' no config ou implemente run() na subclasse."
            )
    
        if hasattr(self._model, "fit_predict"):
            labels = self._model.fit_predict(df)
    
        else:
            self.fit(df)

            coluna_alvo = self.config.get(
                "target_column",
                df.columns[-1]
            )

            X = df.drop(columns=[coluna_alvo])

            labels = self._model.predict(X)
    
        output_column = self.config.get("output_column", "prediction")
    
        df_resultado[output_column] = labels
    
        self.result_ = {
            "output_column": output_column,
            "n_rows": len(df_resultado)
        }
    
        return df_resultado
 
    def summary(self) -> dict:
        """Retorna um dicionário com métricas e resultados estatísticos do modelo."""
        if self.result_ is None:
            return {
                "modelo"    : self.__class__.__name__,
                "config"    : self.config,
                "resultado" : "modelo ainda não executado"
            }
        return {
            "modelo"    : self.__class__.__name__,
            "config"    : self.config,
            "resultado" : self.result_
        }

# fit()            → cada subclasse implementa como quiser (X/y, fit_predict, pipeline com scaler...)
# run()            → retorna o DataFrame do jeito que o modelo precisa
# summary()        → sobrescreve com as métricas específicas (silhouette, F1, AUC...)
# DataPreprocessor → cuida do scaler antes de chegar no modelo
