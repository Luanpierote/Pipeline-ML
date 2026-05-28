""" CARREGA OS DADOS """
# CÓDIGO REFATORADO E FUNCIONAL ✅
# Caso o algoritmo apresente inconsistências, reveja a lógica do compilador Regex responsável por identificar URLs.

# Permite usar "str | Path" como tipo mesmo em outras versões do python
from __future__ import annotations
 
import json  # lê e escreve arquivos .json
import re    # regex para detectar se o path é uma URL
from pathlib import Path   # manipulação de caminhos de pasta/arquivo
from typing import Any     # tipo genérico usado no retorno de profile()
 
import pandas as pd   # monta o DataFrame com os registros carregados
import requests       # faz requisições HTTP para APIs REST
 
# Regex que identifica se uma string é URL:
#   ^https?://          → começa com http:// ou https://
#   ^[^/\\]+\.[^/\\]+/ → ou tem um domínio com ponto antes da primeira barra (ex: site.com/rota)
_IS_URL = re.compile(r"^https?://|^[^/\\]+\.[^/\\]+/")

# Criando pipelines - Em Desenvolvimento⌛
# numeric_transformer = Pipeline(steps = [('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('ohe', OneHotCategoricalEncoder())])
class DataLoader:
  
    # Parameters
    # ----------
    # path     : URL da API ou caminho de pasta local (detectado automaticamente)
    # root_key : chave do JSON com os registros — obrigatório para APIs
    # api_key  : Bearer token, se necessário
    
  def __init__(self, path: str | Path, root_key: str | None = None, api_key: str | None = None):
    raw          = str(path)
    self._is_api = bool(_IS_URL.match(raw)) # True se for URL, False se for pasta
    self.path    = (f"https://{raw}" if not raw.startswith("http") else raw) if self._is_api else Path(raw) # normaliza URL ou converte para Path
    self.root_key, self.api_key, self._df = root_key, api_key, None # armazena configs e inicializa df vazio

  def _get(self, url: str) -> dict | None:
      """Faz GET na URL e retorna o JSON. Retorna None se falhar."""
      headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
      try:
          r = requests.get(url, headers=headers, timeout=10)
          r.raise_for_status()
          return r.json()
      except requests.exceptions.RequestException as e:
          print(f"[DataLoader] {e}"); return None

  
  def _from_api(self, save_folder: Path | None) -> pd.DataFrame:
        """Busca registros da API e, opcionalmente, salva cada um como .json em save_folder."""
        if not self.root_key:
            raise ValueError("root_key obrigatório para APIs.")
        payload = self._get(str(self.path))
        if not payload or self.root_key not in payload:
            raise RuntimeError(f"Chave {self.root_key!r} não encontrada em {self.path!r}.")
        records = payload[self.root_key]
        if save_folder:
            save_folder.mkdir(parents=True, exist_ok=True)
            for r in records:
                (save_folder / f"{r.get('id', hash(json.dumps(r, sort_keys=True)))}.json"
                 ).write_text(json.dumps(r, ensure_ascii=False, indent=2))
        return pd.DataFrame(records)

  def _from_folder(self) -> pd.DataFrame:
        """Lê todos os arquivos .json de uma pasta e retorna um DataFrame."""
        folder = Path(self.path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {folder!r}")
        records = []
        for f in sorted(folder.glob("*.json")):
            try: records.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError) as e: print(f"[DataLoader] Ignorando {f.name}: {e}")
        return pd.DataFrame(records)

  def load(self, save_folder: str | Path | None = None) -> pd.DataFrame:
        """Carrega dados detectando a origem automaticamente."""
        self._df = (
            self._from_api(Path(save_folder) if save_folder else None)
            if self._is_api else self._from_folder()
        )
        return self._df

   
  # precisa retornar tudo que o agente vai usar para tomar decisões. Conectando com cada _decide_*()
  def profile(self) -> dict[str, Any]:
        """Perfil estrutural do DataFrame para agentes de ML."""
        if self._df is None:
            raise RuntimeError("Chame load() antes de profile().")
        df = self._df[[c for c in self._df if not self._df[c].map(lambda x: isinstance(x, (list, dict))).any()]]
        return {
            "n_rows":           int(df.shape[0]),
            "n_cols":           int(df.shape[1]),
            "columns":          df.columns.tolist(),
            "numeric_cols":     df.select_dtypes(include="number").columns.tolist(),
            "categorical_cols": df.select_dtypes(include="object").columns.tolist(),
            "missing_per_col":  df.isnull().sum().to_dict(),
            "unique_per_col":   df.nunique().to_dict(),
            "dtypes":           {c: str(t) for c, t in df.dtypes.items()},
        }

""" TESTE """
if __name__ == "__main__":
    df = DataLoader("dummyjson.com/products", root_key="products").load(save_folder="products")
    print(df.shape)
 
    df2 = DataLoader("products").load()
    print(df2.shape)