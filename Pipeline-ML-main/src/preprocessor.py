"""Pré-processamento dos dados do pipeline de Machine Learning.

Responsável por aplicar as regras definidas no config.json:
- drop_cols
- impute
- preprocessing.encode
- preprocessing.scale
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class Preprocessor:
    """Funções utilitárias de pré-processamento."""

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: list[str] | None, target_column: str | None = None) -> pd.DataFrame:
        """Remove apenas as colunas configuradas e preserva a coluna alvo."""
        if not columns:
            return df.copy()

        cols_to_drop = [col for col in columns if col in df.columns and col != target_column]
        return df.drop(columns=cols_to_drop)

    @staticmethod
    def impute(df: pd.DataFrame, impute_map: dict[str, str] | None) -> pd.DataFrame:
        """Imputa valores nulos por coluna conforme o mapa do config."""
        df = df.copy()
        if not impute_map:
            return df

        for col, strategy in impute_map.items():
            if col not in df.columns:
                continue

            strategy = str(strategy).lower()

            if strategy == "mean":
                df[col] = df[col].fillna(pd.to_numeric(df[col], errors="coerce").mean())
            elif strategy == "median":
                df[col] = df[col].fillna(pd.to_numeric(df[col], errors="coerce").median())
            elif strategy == "mode":
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
            elif strategy == "zero":
                df[col] = df[col].fillna(0)
            else:
                raise ValueError(
                    f"Estratégia de imputação inválida para '{col}': {strategy}. "
                    "Use mean, median, mode ou zero."
                )

        return df

    @staticmethod
    def encode(df: pd.DataFrame, encode_map: dict[str, str] | None, target_column: str | None = None) -> pd.DataFrame:
        """Codifica colunas categóricas com label encoding ou one-hot encoding."""
        df = df.copy()
        if not encode_map:
            return df

        for col, method in encode_map.items():
            if col not in df.columns or col == target_column:
                continue

            method = str(method).lower()

            if method == "label":
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
            elif method == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                raise ValueError(
                    f"Método de encoding inválido para '{col}': {method}. "
                    "Use label ou onehot."
                )

        return df

    @staticmethod
    def scale(df: pd.DataFrame, scale_map: dict[str, str] | None, target_column: str | None = None) -> pd.DataFrame:
        """Escalona colunas numéricas com standard ou minmax."""
        df = df.copy()
        if not scale_map:
            return df

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
        }

        for col, method in scale_map.items():
            if col not in df.columns or col == target_column:
                continue

            method = str(method).lower()
            if method not in scalers:
                raise ValueError(
                    f"Método de escala inválido para '{col}': {method}. "
                    "Use standard ou minmax."
                )

            numeric_col = pd.to_numeric(df[col], errors="coerce")
            if numeric_col.isna().all():
                continue

            df[col] = scalers[method]().fit_transform(numeric_col.to_frame())

        return df


class DataPreprocessor:
    """Executa o pré-processamento usando o contrato oficial do config.json."""

    def __init__(self, config: dict[str, Any] | str | Path):
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("config deve ser dict, str ou Path")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        target_column = self.config.get("target_column")
        preprocessing = self.config.get("preprocessing", {})

        df = Preprocessor.drop_columns(
            df,
            columns=self.config.get("drop_cols", []),
            target_column=target_column,
        )

        df = Preprocessor.impute(
            df,
            impute_map=self.config.get("impute", {}),
        )

        df = Preprocessor.encode(
            df,
            encode_map=preprocessing.get("encode", {}),
            target_column=target_column,
        )

        df = Preprocessor.scale(
            df,
            scale_map=preprocessing.get("scale", {}),
            target_column=target_column,
        )

        return df
