# 5* ETAPA - FUNCIONA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from main import run_agent

""" 
TESTE PARA API
result = run_agent(
    "dummyjson.com/products",
    root_key="products"
) """

result = run_agent(
    "../data/test_dataset.csv"
)


print(result["summary"])
print(result["df_result"].head())