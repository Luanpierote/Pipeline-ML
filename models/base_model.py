"""Classe abstrata base para todos os modelos do pipeline."""
#  Contrato padrão que todo o modelo do pipeline precisa seguir
# Princípio: ela não sabe qual algoritmo irá usar
from abc import ABC, abstractmethod
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
    
            if hasattr(self._model, "predict"):
                labels = self._model.predict(df)
            else:
                raise AttributeError(
                    "[BaseModel] O modelo não possui predict() nem fit_predict()."
                )
    
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
