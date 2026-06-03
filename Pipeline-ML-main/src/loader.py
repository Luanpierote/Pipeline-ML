"""Carregamento de dados para o pipeline.

Suporta três origens:
- URL/API JSON
- pasta local com arquivos .json
- arquivo .csv local
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_IS_URL = re.compile(r"^https?://|^[^/\\]+\.[^/\\]+/")


class DataLoader:
    def __init__(self, source: str | Path, root_key: str | None = None, api_key: str | None = None):
        raw = str(source)
        self._is_api = bool(_IS_URL.match(raw))
        self.source = (f"https://{raw}" if self._is_api and not raw.startswith("http") else raw)
        self.root_key = root_key
        self.api_key = api_key
        self._df: pd.DataFrame | None = None

    def _get(self, url: str) -> dict[str, Any] | list[Any] | None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Erro ao acessar API '{url}': {exc}") from exc

    def _from_api(self, save_folder: Path | None = None) -> pd.DataFrame:
        payload = self._get(str(self.source))

        if self.root_key:
            if not isinstance(payload, dict) or self.root_key not in payload:
                raise RuntimeError(f"Chave root_key={self.root_key!r} não encontrada na resposta da API.")
            records = payload[self.root_key]
        else:
            records = payload

        if not isinstance(records, list):
            raise RuntimeError("A API deve retornar uma lista de registros ou uma chave contendo uma lista.")

        if save_folder:
            save_folder.mkdir(parents=True, exist_ok=True)
            for record in records:
                if isinstance(record, dict):
                    file_name = f"{record.get('id', abs(hash(json.dumps(record, sort_keys=True))))}.json"
                    (save_folder / file_name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        return pd.DataFrame(records)

    def _from_csv(self, path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"CSV não encontrado: {path}")
        return pd.read_csv(path)

    def _from_folder(self, folder: Path) -> pd.DataFrame:
        if not folder.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {folder}")

        records = []
        for file in sorted(folder.glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                records.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[DataLoader] Ignorando {file.name}: {exc}")

        return pd.DataFrame(records)

    def load(self, save_folder: str | Path | None = None) -> pd.DataFrame:
        if self._is_api:
            self._df = self._from_api(Path(save_folder) if save_folder else None)
        else:
            path = Path(self.source)
            if path.suffix.lower() == ".csv":
                self._df = self._from_csv(path)
            else:
                self._df = self._from_folder(path)

        return self._df

    def profile(self) -> dict[str, Any]:
        if self._df is None:
            raise RuntimeError("Chame load() antes de profile().")

        # Remove colunas com listas/dicionários, pois elas quebram modelos sklearn diretamente.
        df = self._df[[
            col for col in self._df.columns
            if not self._df[col].map(lambda value: isinstance(value, (list, dict))).any()
        ]]

        return {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "numeric_cols": df.select_dtypes(include="number").columns.tolist(),
            "categorical_cols": df.select_dtypes(include="object").columns.tolist(),
            "missing_per_col": df.isnull().sum().to_dict(),
            "unique_per_col": df.nunique(dropna=True).to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }


if __name__ == "__main__":
    df = DataLoader("dummyjson.com/products", root_key="products").load(save_folder="products")
    print(df.shape)
