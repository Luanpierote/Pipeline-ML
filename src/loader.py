"""
CARREGA OS DADOS

Suporta:
- APIs REST (JSON)
- Pastas contendo JSONs
- Arquivos CSV
- Arquivos Excel (.xlsx/.xls)
- Arquivos Parquet

Retorna sempre um DataFrame pandas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# Detecta URLs
_IS_URL = re.compile(r"^https?://|^[^/\\]+\.[^/\\]+/")


class DataLoader:

    def __init__(
        self,
        path: str | Path,
        root_key: str | None = None,
        api_key: str | None = None
    ):

        raw = str(path)

        self._is_api = bool(_IS_URL.match(raw))

        if self._is_api:
            self.path = (
                raw
                if raw.startswith("http")
                else f"https://{raw}"
            )
        else:
            self.path = Path(raw)

        self.root_key = root_key
        self.api_key = api_key

        self._df = None

    # --------------------------------------------------
    # API
    # --------------------------------------------------

    def _get(self, url: str) -> dict | None:

        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else {}
        )

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:

            print(f"[DataLoader] {e}")

            return None

    def _from_api(self, save_folder: Path | None) -> pd.DataFrame:
        print("ROOT_KEY =", self.root_key)

        payload = self._get(str(self.path))

        if not payload:
            raise RuntimeError(
                f"Falha ao obter dados de {self.path}"
            )

        if self.root_key:

            if self.root_key not in payload:
                raise RuntimeError(
                    f"Chave '{self.root_key}' não encontrada."
                )

            records = payload[self.root_key]

        elif isinstance(payload, list):

            records = payload

        elif isinstance(payload, dict):

            list_keys = [
                k
                for k, v in payload.items()
                if isinstance(v, list)
            ]

            if len(list_keys) == 1:

                records = payload[list_keys[0]]

            elif len(list_keys) > 1:

                raise ValueError(
                    f"API possui múltiplas listas: {list_keys}. "
                    "Informe root_key."
                )

            else:

                raise ValueError(
                    "Nenhuma lista encontrada na resposta da API."
                )

        else:

            raise ValueError(
                "Formato de resposta não suportado."
            )

        if save_folder:

            save_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            for record in records:

                file_id = record.get(
                    "id",
                    hash(
                        json.dumps(
                            record,
                            sort_keys=True
                        )
                    )
                )

                filepath = save_folder / f"{file_id}.json"

                filepath.write_text(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        indent=2
                    )
                )

        return pd.DataFrame(records)

    # --------------------------------------------------
    # PASTA JSON
    # --------------------------------------------------

    def _from_folder(self) -> pd.DataFrame:

        records = []

        for file in sorted(self.path.glob("*.json")):

            try:

                records.append(
                    json.loads(file.read_text())
                )

            except (json.JSONDecodeError, OSError) as e:

                print(
                    f"[DataLoader] Ignorando "
                    f"{file.name}: {e}"
                )

        return pd.DataFrame(records)

    # --------------------------------------------------
    # ARQUIVOS TABULARES
    # --------------------------------------------------

    def _from_file(self) -> pd.DataFrame:

        suffix = self.path.suffix.lower()

        if suffix == ".csv":

            return pd.read_csv(self.path)

        if suffix in (".xlsx", ".xls"):

            return pd.read_excel(self.path)

        if suffix == ".parquet":

            return pd.read_parquet(self.path)

        raise ValueError(
            f"Formato não suportado: {suffix}"
        )

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def load(
        self,
        save_folder: str | Path | None = None
    ) -> pd.DataFrame:

        if self._is_api:

            self._df = self._from_api(
                Path(save_folder)
                if save_folder
                else None
            )

        else:

            path = Path(self.path)

            if path.is_dir():

                self._df = self._from_folder()

            elif path.is_file():

                self._df = self._from_file()

            else:

                raise FileNotFoundError(
                    f"Caminho não encontrado: {path}"
                )

        return self._df

    # --------------------------------------------------
    # PROFILE
    # --------------------------------------------------

    def profile(self) -> dict[str, Any]:

        if self._df is None:

            raise RuntimeError(
                "Execute load() antes de profile()."
            )

        df = self._df.copy()

        # remove colunas contendo listas/dicionários
        valid_cols = []

        for col in df.columns:

            has_nested = df[col].map(
                lambda x: isinstance(
                    x,
                    (list, dict)
                )
            ).any()

            if not has_nested:
                valid_cols.append(col)

        df = df[valid_cols]

        return {

            "n_rows":
                int(df.shape[0]),

            "n_cols":
                int(df.shape[1]),

            "columns":
                df.columns.tolist(),

            "numeric_cols":
                df.select_dtypes(
                    include="number"
                ).columns.tolist(),

            "categorical_cols":
                df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist(),

            "missing_per_col":
                df.isnull().sum().to_dict(),

            "unique_per_col":
                df.nunique().to_dict(),
            # Identificação do tipo de cada variável
            "dtypes":
                {
                    col: str(dtype)
                    for col, dtype
                    in df.dtypes.items()
                },
            # Colunas que foram descartadas
            "dropped_nested": [col for col in self._df.columns if col not in valid_cols]
        }


# --------------------------------------------------
# TESTE
# --------------------------------------------------

if __name__ == "__main__":

    # API

    df = DataLoader(
        "dummyjson.com/products",
        root_key="products"
    ).load(
        save_folder="products"
    )

    print(df.shape)

    # Pasta JSON

    df2 = DataLoader(
        "products"
    ).load()

    print(df2.shape)
