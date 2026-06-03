"""
Pipeline runner — orquestra todas as etapas do sistema.
 
Fluxo:
    CONFIG  →  loader  →  validator  →  preprocessor  →  modelo  →  output
"""
# from sklearn.pipeline import Pipeline
from __future__ import annotations
 
import json
from pathlib import Path
from typing import Any
 
import pandas as pd
 
import importlib
 
from src.loader        import DataLoader
from src.validator     import DataValidator
from src.preprocessor  import DataPreprocessor
 
 
 
class PipelineRunner:
    """
    Orquestra as etapas: carga → validação → pré-processamento → modelo.
 
    Parameters
    ----------
    config : dict | str | Path
        Dicionário de configuração ou caminho para o config.json.
        Se for uma lista (múltiplos datasets), use PipelineRunner.run_all().
    verbose : bool
        Exibe logs de cada etapa quando True.
    """
 
    def __init__(self, config: dict | str | Path, verbose: bool = True):
        self.config  = self._load_config(config)
        self.verbose = verbose
 
    @staticmethod
    def _load_config(config: dict | str | Path) -> dict:
        if isinstance(config, dict):
            return config
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrado: {path}")
        with open(path) as f:
            return json.load(f)
 
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
 
    def _load(self) -> pd.DataFrame:
        # CONTRATO DIVERGENTE: o runner espera as chaves "source", "root_key",
        # "api_key", "save_folder" e "name" no config — mas o MLAgent só salva
        # "drop_cols", "impute", "encode", "scale" e "model".
        # Solução: MLAgent.__init__ deve receber o path/URL de origem e
        # incluí-lo no config.json sob a chave "source".
        source      = self.config.get("source")
        root_key    = self.config.get("root_key")
        api_key     = self.config.get("api_key")
        save_folder = self.config.get("save_folder")
 
        if not source:
            raise ValueError("Config deve conter a chave 'source' com o caminho/URL dos dados.")
 
        self._log(f"\n[1/4] Carregando dados de: {source}")
        loader = DataLoader(source, root_key=root_key, api_key=api_key)
        df = loader.load(save_folder=save_folder)
        self._log(f"       Shape carregado: {df.shape}")
        return df
 
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("\n[2/4] Validando dados...")
        validator = DataValidator(df, self.config)
        result    = validator.validate()
        result.summary()
 
        if not result.is_valid:
            raise RuntimeError(
                f"Validação falhou com {len(result.errors)} erro(s). "
                "Corrija os dados ou o config antes de continuar."
            )
        return df
 
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("\n[3/4] Pré-processando...")
        preprocessor = DataPreprocessor(self.config)
        df_clean = preprocessor.run(df)
        self._log(f"       Shape após pré-processamento: {df_clean.shape}")
        return df_clean

#  Agora qualquer string "sklearn.cluster.DBSCAN" ou "sklearn.ensemble.RandomForestClassifier" funciona automaticamente, sem precisar criar DBSCANModel, RandomForestModel ou qualquer subclasse.
    def _run_model(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        model_cfg = self.config.get("model")
        if not model_cfg:
            raise ValueError("Config deve conter a chave 'model' com 'model_class' e 'model_params'.")

        model_class_path = model_cfg.get("model_class")
        model_params     = model_cfg.get("model_params", {})

        if not model_class_path:
            raise ValueError("'model' deve conter 'model_class' (ex: 'sklearn.cluster.DBSCAN').")

        self._log(f"\n[4/4] Executando modelo: {model_class_path}")

        # importlib — sem registry, sem subclasses
        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)

        model  = ModelClass(**model_params)
        df_out = model.fit_predict(df)          # API sklearn padrão
        # summary genérico — adapte conforme o modelo
        summary = {"model": model_class_path, "params": model_params}

        return pd.DataFrame({"label": df_out}, index=df.index), summary
 
    @staticmethod
    def run(self) -> dict[str, Any]:
        """
        Executa o pipeline completo para um único dataset.
 
        Returns
        -------
        dict com:
            df_result  : DataFrame com predições/rótulos do modelo
            summary    : métricas e metadados do modelo
            config     : config usado nesta execução
        """
        name = self.config.get("name", "dataset")
        self._log(f"\n{'='*50}\n  PIPELINE: {name.upper()}\n{'='*50}")
 
        df        = self._load()
        df        = self._validate(df)
        df_clean  = self._preprocess(df)
        df_result, summary = self._run_model(df_clean)
 
        self._log(f"\nPipeline '{name}' concluído com sucesso.\n")
        return {"df_result": df_result, "summary": summary, "config": self.config}
 
    @classmethod
    def run_all(cls, config_path: str | Path, verbose: bool = True) -> list[dict[str, Any]]:
        """
        Executa o pipeline para cada dataset listado no config.json.
        O config.json pode ser um único objeto ou uma lista de objetos.
        """
        path = Path(config_path)
        with open(path) as f:
            raw = json.load(f)
 
        configs = raw if isinstance(raw, list) else [raw]
        results = []
 
        for cfg in configs:
            runner = cls(cfg, verbose=verbose)
            try:
                results.append(runner.run())
            except Exception as e:
                # TODO: run_agent() em src/agent.py ainda não instancia o PipelineRunner
                # nem chama runner.run() — a função retorna None em vez do resultado
                name = cfg.get("name", "?")
                print(f"\nPipeline '{name}' falhou: {e}")
                results.append({"config": cfg, "error": str(e)})
 
        return results