"""Pipeline runner — orquestra carga, validação, pré-processamento e modelo."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.validator import DataValidator


class PipelineRunner:
    def __init__(self, config: dict[str, Any] | str | Path, verbose: bool = True):
        self.config = self._load_config(config)
        self.verbose = verbose

    @staticmethod
    def _load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
        if isinstance(config, dict):
            return config

        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrado: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _load(self) -> pd.DataFrame:
        source = self.config.get("source")
        if not source or source == "...":
            raise ValueError("Config deve conter a chave 'source' com caminho/URL válido dos dados.")

        self._log(f"\n[1/4] Carregando dados de: {source}")
        loader = DataLoader(
            source,
            root_key=self.config.get("root_key"),
            api_key=self.config.get("api_key"),
        )
        df = loader.load(save_folder=self.config.get("save_folder"))
        self._log(f"       Shape carregado: {df.shape}")
        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("\n[2/4] Validando dados...")
        result = DataValidator(df, self.config).validate()
        result.summary()

        if not result.is_valid:
            raise RuntimeError(
                f"Validação falhou com {len(result.errors)} erro(s). "
                "Corrija os dados ou o config antes de continuar."
            )
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("\n[3/4] Pré-processando...")
        df_clean = DataPreprocessor(self.config).run(df)
        self._log(f"       Shape após pré-processamento: {df_clean.shape}")
        return df_clean

    def _resolve_model(self):
        model_cfg = self.config.get("model", {})
        model_class_path = model_cfg.get("model_class")
        model_params = model_cfg.get("model_params", {})

        if not model_class_path or model_class_path == "...":
            raise ValueError("Config deve conter model.model_class válido.")

        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class(**model_params), model_class_path, model_params

    @staticmethod
    def _prepare_features(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        features = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df.copy()

        # Garante entrada numérica para modelos sklearn comuns.
        features = pd.get_dummies(features, dummy_na=False)
        features = features.apply(pd.to_numeric, errors="coerce")
        features = features.fillna(0)
        return features

    def _run_model(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        model, model_class_path, model_params = self._resolve_model()
        target_column = self.config.get("target_column")
        output_column = self.config.get("output_column", "prediction")

        self._log(f"\n[4/4] Executando modelo: {model_class_path}")

        X = self._prepare_features(df, target_column)

        if target_column and target_column in df.columns:
            y = df[target_column]
            model.fit(X, y)
            predictions = model.predict(X)
            task = "supervised"
        elif hasattr(model, "fit_predict"):
            predictions = model.fit_predict(X)
            task = "unsupervised"
        else:
            model.fit(X)
            predictions = model.predict(X)
            task = "unsupervised"

        df_result = df.copy()
        df_result[output_column] = predictions

        summary = {
            "model": model_class_path,
            "params": model_params,
            "task": task,
            "output_column": output_column,
            "n_rows": int(len(df_result)),
        }

        return df_result, summary

    def run(self) -> dict[str, Any]:
        name = self.config.get("name", "dataset")
        self._log(f"\n{'=' * 50}\n  PIPELINE: {str(name).upper()}\n{'=' * 50}")

        df = self._load()
        df = self._validate(df)
        df_clean = self._preprocess(df)
        df_result, summary = self._run_model(df_clean)

        self._log(f"\nPipeline '{name}' concluído com sucesso.\n")
        return {"df_result": df_result, "summary": summary, "config": self.config}

    @classmethod
    def run_all(cls, config_path: str | Path, verbose: bool = True) -> list[dict[str, Any]]:
        path = Path(config_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        configs = raw if isinstance(raw, list) else [raw]
        results = []

        for cfg in configs:
            runner = cls(cfg, verbose=verbose)
            try:
                results.append(runner.run())
            except Exception as exc:
                name = cfg.get("name", "?")
                print(f"\nPipeline '{name}' falhou: {exc}")
                results.append({"config": cfg, "error": str(exc)})

        return results
