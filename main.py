""" EXIBE O RESULTADO FINAL(CARACTERÍSTICAS DO DATASET E MÉTRICAS DO MODELO) DE FORMA ESTRUTURADA """
"""
MLAgent
--------
Analisa um dataset e gera automaticamente toda a especificação
do pipeline:

- validação
- pré-processamento
- modelo
- hiperparâmetros

O PipelineRunner apenas executa o config produzido aqui.
"""

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
        self.profile = {}
        # config final — preenchido pelo configure()
        self.config = {}

    # ---------------------------------------------------------
    # ETAPA 1
    # ---------------------------------------------------------

    def analyze(self):
        """
        Carrega o dataset e produz um perfil estatístico.
        O perfil é o que o agente usa para tomar todas as decisões.
        """

        loader = DataLoader(self.csv_path)
        loader.load()

        self.profile = loader.profile()

    # ---------------------------------------------------------
    # ETAPA 2
    # ---------------------------------------------------------

    def configure(self):
        """
        Etapa 2 — usa o perfil para decidir cada etapa do preprocessing.
        Grava as decisões no config.json.
        """
        model_cfg = self._decide_model()

        self.config = {

            # =====================================
            # ORIGEM
            # =====================================

            "source": self.csv_path,

            # =====================================
            # VALIDAÇÃO
            # =====================================

            "required": self._decide_required(),

            "critical": self._decide_critical(),

            "unique": self._decide_unique(),

            "ranges": self._decide_ranges(),

            "dtypes": self._decide_dtypes(),

            # =====================================
            # PREPROCESSAMENTO
            # =====================================

            "drop_cols": self._decide_drop(),

            "impute": self._decide_impute(),

            "encode": self._decide_encode(),

            "scale": self._decide_scale(),

            # =====================================
            # MODELAGEM
            # =====================================

            "target_column": self._decide_target(),

            "model": self._decide_model()   # ← devolve o dict inteiro, aninhado
        }

        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    # ---------------------------------------------------------
    # VALIDAÇÃO
    # ---------------------------------------------------------

    def _decide_required(self):

        return list(self.profile.get("columns", []))

    def _decide_critical(self):

        # colunas numéricas sem nenhum nulo são provavelmente essenciais
        return [
            col for col in self.profile["numeric_cols"]
            if self.profile["missing_per_col"].get(col, 0) == 0
        ]

    def _decide_unique(self):

        n_rows = self.profile["n_rows"]

        return [
            col
            for col, unique_count
            in self.profile["unique_per_col"].items()
            if unique_count == n_rows
        ]

    def _decide_ranges(self):

        ranges = {}

        for col in self.profile["numeric_cols"]:

            name = col.lower()

            if "price" in name:
                ranges[col] = [0, None]

            elif "rating" in name:
                ranges[col] = [0, 5]

            elif "age" in name:
                ranges[col] = [0, 120]

            elif "stock" in name:
                ranges[col] = [0, None]

        return ranges

    def _decide_dtypes(self):

        return {
            col: str(dtype)
            for col, dtype
            in self.profile["dtypes"].items()
        }

    # ---------------------------------------------------------
    # PREPROCESSAMENTO
    # ---------------------------------------------------------

    def _decide_drop(self):
        """
        Função de entrada — chamada diretamente no Colab.
        Instancia o agente, analisa, configura e dispara o runner.
        """

        n_rows = self.profile["n_rows"]

        return [
            col
            for col, unique_count
            in self.profile["unique_per_col"].items()
            if "id" in col.lower()
            or (unique_count / n_rows) >= 0.90
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

        drop_cols = self._decide_drop()

        return {

            col: (
                "label"
                if self.profile["unique_per_col"][col] <= 10
                else "onehot"
            )

            for col in self.profile["categorical_cols"]
            if col not in drop_cols
        }

    def _decide_scale(self):
        """
        Decide qual scaler usar nas colunas numéricas.
        Regra: todas as numéricas que não foram dropadas → standard
        """

        drop_cols = self._decide_drop()

        return {

            col: "standard"

            for col in self.profile["numeric_cols"]

            if col not in drop_cols
        }

    # ---------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------

    def _decide_target(self):

        candidate_targets = [

            col

            for col in self.profile["categorical_cols"]

            if self.profile["unique_per_col"][col]
            < self.profile["n_rows"] * 0.20
        ]

        if candidate_targets:
            return candidate_targets[0]

        return None

    # ---------------------------------------------------------
    # MODELO
    # ---------------------------------------------------------

    def _decide_model(self):
        """
        Decide qual modelo usar com base no perfil do dataset.
        Regra: sem coluna alvo clara → clustering (DBSCAN)
        """
        target = self._decide_target()

        if target is None:

            return {

                "model_class":
                    "sklearn.cluster.DBSCAN",

                "model_params": {

                    "eps": 0.5,
                    "min_samples": 5
                }
            }

        return {

            "model_class":
                "sklearn.ensemble.RandomForestClassifier",

            "model_params": {

                "n_estimators": 100,
                "random_state": 42
            }
        }

    # ---------------------------------------------------------
    # EXECUÇÃO
    # ---------------------------------------------------------

    def run(self):
        """Analisa, configura e dispara o PipelineRunner."""
        self.analyze()

        self.configure()  # salva config.json em disco

        runner = PipelineRunner(self.config) 

        return runner.run() # retorna {"df_result", "summary", "config"}


# ---------------------------------------------------------
# FUNÇÃO DE ENTRADA
# ---------------------------------------------------------

def run_agent(csv_path: str):
    """
    Função de entrada — chamada diretamente no Colab.
    Instancia o agente, analisa, configura e dispara o runner.
    """
    # instancia o agente
    # chama analyze()
    # chama configure()
    # instancia o PipelineRunner com o config.json gerado
    # chama runner.run()
    # retorna o resultado
    agent = MLAgent(csv_path)

    return agent.run() 


# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------

if __name__ == "__main__":

    result = run_agent("dataset.csv")

    print(result["summary"])