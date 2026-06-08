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
É o agente quem coordena todas as atividades do sistema.
"""

import json

from src.loader import DataLoader
from pipeline.runner import PipelineRunner

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

#    Agente heurístico que analisa o dataset e decide automaticamente
#     como configurar o pipeline — sem intervenção humana. 
    
class MLAgent:

    def __init__(self, source,root_key = None, api_key = None):
         # caminho da fonte de dados que o agente vai analisar(seja Excel,csv,parquet,...)
        self.source = source
        self.root_key = root_key
        self.api_key = api_key
        
         # perfil do dataset — preenchido pelo analyze()
        self.profile = {}
        # config final — preenchido pelo configure()
        self.config = {}

    
    # ETAPA 1
    # ---------------------------------------------------------

    def analyze(self):
        """
        Carrega o dataset e produz um perfil estatístico.
        O perfil é o que o agente usa para tomar todas as decisões.
        """

        loader = DataLoader(
            self.source,
            root_key=self.root_key,
            api_key=self.api_key
        )

        loader.load()

        self.profile = loader.profile()

   
    # ETAPA 2
    # ---------------------------------------------------------

    def configure(self):
        """
        Etapa 2 — usa o perfil para decidir cada etapa do preprocessing.
        Grava as decisões no config.json.
        """
        model_cfg = self._decide_model()

        self.config = {

            
            # ORIGEM
           

            "source": self.source,

    
            # VALIDAÇÃO
           

            "required": self._decide_required(),

            "critical": self._decide_critical(),

            "unique": self._decide_unique(),

            "ranges": self._decide_ranges(),

            "dtypes": self._decide_dtypes(),

          
            # PREPROCESSAMENTO
            

            "drop_cols": self._decide_drop(),

            "impute": self._decide_impute(),

            "encode": self._decide_encode(),

            "scale": self._decide_scale(),

            
            # MODELAGEM
            

            "target_column": self._decide_target(),

            "model": self._decide_model()   # ← devolve o dict inteiro, aninhado
        }
        
        # Salva sempre no config presente na raiz do projeto 
        configs_dir = ROOT_DIR / "configs"
        configs_dir.mkdir(exist_ok=True)

        dataset_name = Path(self.source).stem

        config_path = configs_dir / f"{dataset_name}.json"

        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

   
    # VALIDAÇÃO
    

    def _decide_required(self):

        return list(self.profile.get("columns", []))

    def _decide_critical(self):

        # colunas numéricas sem nenhum nulo são provavelmente essenciais
        return [
            col for col in self.profile["numeric_cols"]
            if self.profile["missing_per_col"].get(col, 0) == 0
        ]

    # Infere identificadores únicos no dataset automaticamente
    def _decide_unique(self):

        n_rows = self.profile["n_rows"]

        unique_cols = []

        for col, unique_count in self.profile["unique_per_col"].items():

            if unique_count == n_rows:

                nome = col.lower()

                if any(
                    token in nome
                    for token in [
                        "id",
                        "uuid",
                        "sku",
                        "key",
                        "code"
                    ]
                ):
                    unique_cols.append(col)

        return unique_cols

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

    # PREPROCESSAMENTO
    # ---------------------------------------------------------

    def _decide_drop(self):
        """
        Função de entrada — chamada diretamente no Colab.
        Instancia o agente, analisa, configura e dispara o runner.
        """
        n_rows = max(self.profile["n_rows"], 1)
        
        drop_cols = []

        for col in self.profile["columns"]:
            if "id" in col.lower():
                drop_cols.append(col)

        for col in self.profile["categorical_cols"]:

            unique_ratio = (
                self.profile["unique_per_col"][col]
                / n_rows
                )

            # Problema: Caso o número de colunas seja igual ao número de registros, o valor atingirá >= 0.90 e o código poderá dropar features importantes
            # Segue o princípio da cardinalidade para relacionar os dados - Colunas categoricas com cardinalidade muito alta são removidas(ele analisa: quantos valores diferentes uma coluna tem?)
            if unique_ratio > 0.95: 
                drop_cols.append(col)
        

        drop_cols.extend(
            self.profile.get("dropped_nested", [])
        )

        return list(set(drop_cols))

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

    # Escolhe colunas categoricas alvos para os modelos de predição, desconsiderando as categoricas irrelevantes
    def _decide_target(self):
        target_keywords = [
            "target",
            "label",
            "class",
            "category",
            "prediction",
            "output"
        ]

          # 1. Procura nomes sugestivos
        for col in self.profile["columns"]:
            if any(k in col.lower() for k in target_keywords):
                return col

        n_rows = max(self.profile["n_rows"], 1)

        # 2. Procura categóricas de baixa cardinalidade
        categorical_candidates = [
            col
            for col in self.profile["categorical_cols"]
            if self.profile["unique_per_col"][col] / n_rows < 0.20
        ]

        if categorical_candidates:
            return min(
                categorical_candidates,
                key=lambda c: self.profile["unique_per_col"][c]
            )

        # 3. Procura targets numéricos para regressão
        numeric_candidates = [
            col
            for col in self.profile["numeric_cols"]
            if "id" not in col.lower()
            and self.profile["unique_per_col"][col] / n_rows < 0.50
        ]

        if numeric_candidates:
            return numeric_candidates[-1]

        return None

    # ---------------------------------------------------------
    # MODELO
    # ---------------------------------------------------------

    def _decide_model(self) -> dict:
        """
        Heurística em camadas — decide o estimador sklearn mais adequado
        com base no perfil do dataset, sem nenhuma intervenção humana.
    
        Camada 1 — supervisionado vs não-supervisionado (tem alvo?)
        Camada 2 — regressão vs classificação (alvo contínuo ou discreto?)
        Camada 3 — porte do dataset (linhas, colunas, esparsidade)
        """
        target   = self._decide_target()
        n_rows   = self.profile["n_rows"]
        n_cols   = self.profile["n_cols"]
    
        # ── Camada 1: sem alvo → não-supervisionado ────────────────────────
        if target is None:
            return self._decide_unsupervised(n_rows, n_cols)
    
        # ── Camada 2: tem alvo → supervisionado ───────────────────────────
        n_unique_target = self.profile["unique_per_col"][target]
        target_dtype    = self.profile["dtypes"][target]
        is_continuous   = (
            "float" in target_dtype                        # dtype float → contínuo
            or n_unique_target / n_rows > 0.15             # muitos valores únicos → contínuo
        )
    
        if is_continuous:
            return self._decide_regressor(n_rows, n_cols, target)
        else:
            return self._decide_classifier(n_rows, n_cols, n_unique_target, target)
    
    
    # ── Submétodos por família ─────────────────────────────────────────────
    
    def _decide_unsupervised(self, n_rows: int, n_cols: int) -> dict:
        """
        Sem coluna alvo.
        - Muitas linhas + poucas colunas  → KMeans (rápido, escalável)
        - Poucas linhas ou alta dimensão  → DBSCAN (não precisa definir k)
        - Dataset muito pequeno           → IsolationForest (detecção de anomalia)
        """
        if n_rows < 500:
            return {
                "model_class":  "sklearn.ensemble.IsolationForest",
                "model_params": { "contamination": 0.05, "random_state": 42 },
                "target": None
            }
        if n_cols <= 20 and n_rows >= 5000:
            return {
                "model_class":  "sklearn.cluster.KMeans",
                "model_params": { "n_clusters": 8, "random_state": 42, "n_init": "auto" },
                "target": None
            }
        return {
            "model_class":  "sklearn.cluster.DBSCAN",
            "model_params": { "eps": 0.5, "min_samples": 5 },
            "target": None
        }
    
    
    def _decide_regressor(self, n_rows: int, n_cols: int, target: str) -> dict:
        """
        Alvo contínuo.
        - Dataset grande (>5k)   → GradientBoostingRegressor (melhor acurácia)
        - Dataset pequeno        → Ridge (estável com poucos dados)
        - Alta dimensão (>50 col)→ Lasso (seleção automática de features)
        """
        if n_cols > 50:
            return {
                "model_class":  "sklearn.linear_model.Lasso",
                "model_params": { "alpha": 1.0 },
                "target": target
            }
        if n_rows >= 5000:
            return {
                "model_class":  "sklearn.ensemble.GradientBoostingRegressor",
                "model_params": { "n_estimators": 100, "random_state": 42 },
                "target": target
            }
        return {
            "model_class":  "sklearn.linear_model.Ridge",
            "model_params": { "alpha": 1.0 },
            "target": target
        }
    
    
    def _decide_classifier(self, n_rows: int, n_cols: int,
                            n_unique_target: int, target: str) -> dict:
        """
        Alvo discreto.
        - Muitas classes (>10)   → LogisticRegression (generaliza melhor)
        - Dataset grande (>5k)   → GradientBoostingClassifier (alta performance)
        - Dataset pequeno        → RandomForestClassifier (robusto com poucos dados)
        - Alta dimensão (>50 col)→ SVC com kernel linear
        """
        if n_cols > 50:
            return {
                "model_class":  "sklearn.svm.SVC",
                "model_params": { "kernel": "linear", "probability": True },
                "target": target
            }
        if n_unique_target > 10:
            return {
                "model_class":  "sklearn.linear_model.LogisticRegression",
                "model_params": { "max_iter": 1000, "random_state": 42 },
                "target": target
            }
        if n_rows >= 5000:
            return {
                "model_class":  "sklearn.ensemble.GradientBoostingClassifier",
                "model_params": { "n_estimators": 100, "random_state": 42 },
                "target": target
            }
        return {
            "model_class":  "sklearn.ensemble.RandomForestClassifier",
            "model_params": { "n_estimators": 100, "random_state": 42 },
            "target": target
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

def run_agent(source, root_key=None, api_key=None):
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
    agent = MLAgent(
        source,
        root_key=root_key,
        api_key=api_key
    )

    return agent.run() 


# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------

if __name__ == "__main__":

    result = run_agent("dataset.csv")

    print(result["summary"])