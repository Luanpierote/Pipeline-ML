# 2* ETAPA - Funcionando
import sys
from pathlib import Path
import pandas as pd
import json
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.validator import DataValidator
from main import MLAgent

agent = MLAgent("../data/test_dataset.csv")
agent.analyze()
agent.configure()

df = pd.read_csv("../data/test_dataset.csv")

result = DataValidator(df,agent.config).validate()
result.summary()
# esperado: ✅ VÁLIDO — se quebrar aqui o problema é no _decide_required/_decide_ranges

# Copia do dataframe para testar exceções que violam as regras
df_teste = df.copy()

# Testando coluna obrigatória faltando : Funciona
df_teste = df.drop(columns=["price"])

# Testando valor fora do range : Funciona
df_teste.loc[0, "rating"] = 999

# Testando se ele irá sinalizar a coluna crítica nula :  Funciona
df_teste.loc[0, "price"] = None

# Testando Dtype incorreto : funciona
df_teste["price"] = df_teste["price"].astype(str)
print(df_teste.dtypes)
# Coluna com valor negativo : funciona
df_teste.loc[0, "price"] = -10


result_test = DataValidator(df_teste,agent.config).validate()
# print("Status:", result_test.is_valid())
result_test.summary()
print(result_test.errors)
# Warnings só é acionado quado existem duplicadas encontradas nas colunas do dataframe
print(result_test.warnings)