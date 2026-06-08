# 4* ETAPA - Funcionando
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.preprocessor import DataPreprocessor
from main import MLAgent


"""
 TESTE PARA API
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

from models.base_model import BaseModel

# Printa os datatypes de cada coluna
print(df_clean.select_dtypes(include="object").columns)

model = BaseModel(agent.config)
df_result = model.run(df_clean)

print(df_result.columns)   # esperado: coluna "cluster" ou "prediction"
print(model.summary())