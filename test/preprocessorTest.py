# 3* ETAPA - Funcionando
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.preprocessor import DataPreprocessor
from main import MLAgent

"""
PARA API 
agent = MLAgent(
    "dummyjson.com/products",
    root_key="products"
) """

agent = MLAgent(
    "../data/test_dataset.csv"
)

agent.analyze()
agent.configure()

df = pd.read_csv("../data/test_dataset.csv")

df_clean = DataPreprocessor(agent.config).run(df)

print(df_clean.shape)
print(df_clean.dtypes)
# verificar: colunas dropadas sumiram, sem nulos, categóricas encodadas, numéricas escaladas