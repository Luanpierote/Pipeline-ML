""" EXIBE O RESULTADO FINAL(CARACTERÍSTICAS DO DATASET E MÉTRICAS DO MODELO) DE FORMA ESTRUTURADA """
# EM DESENVOLVIMENTO - Pronto, porém, demanda testes⌛
import json
from src.loader import DataLoader
from pipeline.runner import PipelineRunner
  
#    Agente heurístico que analisa o dataset e decide automaticamente
#     como configurar o pipeline — sem intervenção humana. 
    
class MLAgent:

    def __init__(self, csv_path: str):
        # caminho do CSV que o agente vai analisar
        self.csv_path = csv_path
        # perfil do dataset — preenchido pelo analyze()
        self.profile  = {}
        # config final — preenchido pelo configure()
        self.config   = {}
        pass

    def analyze(self):
        """
        Etapa 1 — carrega o dataset e extrai o perfil.
        O perfil é o que o agente usa para tomar todas as decisões.
        """
    
        # carrega o CSV usando o DataLoader
        loader = DataLoader(self.csv_path)
        loader.load()
        # guarda o resultado em self.profile
        self.profile = loader.profile() # n_rows, colunas, nulos, unicidade, dtypes..
    

    def configure(self):
        """
        Etapa 2 — usa o perfil para decidir cada etapa do preprocessing.
        Grava as decisões no config.json.
        """
        self.config = {
        # chama _decide_drop()
        "drop_cols": self._decide_drop(),
        # chama _decide_impute()
        "impute": self._decide_impute(),
        # chama _decide_encode()
        "encode": self._decide_encode(),
        # chama _decide_scale()
        "scale": self._decide_scale(),
        # chama _decide_model()
        "model": self._decide_model()
        }
        # monta o dicionário self.config com todas as decisões
        with open("config.json","w")as f:
            # Salva o dicionário como texto JSON em um arquivo e grava o config.json em disco
            json.dump(self.config, f, indent=2)
        
    
    def _decide_drop(self) -> list[str]:  # Indica que a função vai devolver uma lista de strings
        """
        Decide quais colunas remover.
        Regra: colunas com nome contendo 'id' ou unicidade >= 90%
        """
        n = self.profile["n_rows"]
        return [
            col for col, unique in self.profile["unique_per_col"].items()
            if "id" in col.lower() or (unique / n) >= 0.9
        ]

    def _decide_impute(self):
        """
        Decide como preencher nulos por coluna.
        Regra: numéricas → median | categóricas → mode
        """
        strategy = {}
        for col in self.profile["numeric_cols"]:
            if self.profile["missing_per_col"].get(col, 0) > 0:
                strategy[col] = "median"
        for col in self.profile["categorical_cols"]:
            if self.profile["missing_per_col"].get(col, 0) > 0:
                strategy[col] = "mode"
        return strategy

    def _decide_encode(self):
        """
        Decide qual encoder usar por coluna categórica.
        Regra: <= 10 categorias únicas → label | > 10 → onehot
        """
        return {
            col: ("label" if self.profile["unique_per_col"][col] <= 10 else "onehot")
            for col in self.profile["categorical_cols"]
            if col not in self._decide_drop()
        }

    def _decide_scale(self):
        """
        Decide qual scaler usar nas colunas numéricas.
        Regra: todas as numéricas que não foram dropadas → standard
        """
        drop = self._decide_drop()
        return {
            col: "standard"
            for col in self.profile["numeric_cols"]
            if col not in drop
        }

    def _decide_model(self):
        """
        Decide qual modelo usar com base no perfil do dataset.
        Regra: sem coluna alvo clara → clustering (DBSCAN)
        """
        max_unique = max(self.profile["unique_per_col"].values(), default=0)

        if max_unique / self.profile["n_rows"] < 0.05:
            return {
                "model_class":  "sklearn.cluster.DBSCAN",
                "model_params": { "eps": 0.5, "min_samples": 5 }
            }
        return {
            "model_class":  "sklearn.ensemble.RandomForestClassifier",
            "model_params": { "n_estimators": 100, "random_state": 42 }
        }


def run(self) -> dict:
    """Executa o pipeline completo e retorna o config gerado."""
    self.analyze()
    self.configure()
    return self.config

def run_agent(csv_path: str) -> dict:
    """
    Função de entrada — chamada diretamente no Colab.
    Instancia o agente, analisa, configura e dispara o runner.
    """
    """Ponto de entrada — instancia o agente, analisa, configura e retorna o resultado."""
    # instancia o agente
    # chama analyze()
    # chama configure()
    # instancia o PipelineRunner com o config.json gerado
    # chama runner.run()
    # retorna o resultado
    return MLAgent(csv_path).run()

if __name__ == "__main__":
    # Roda todos os datasets listados no config.json
    results = PipelineRunner.run_all("config.json")
 
    for r in results:
        if "error" not in r:
            df = r["df_result"]
            print(f"\nDataset : {r['config'].get('name')}")
            print(f"Shape   : {df.shape}")
            print(f"Resumo  : {r['summary']}")
            print(df.head())

    
   
    