# 1* ETAPA - FUNCIONA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.loader import DataLoader

loader = DataLoader("../data/test_dataset.csv")

df = loader.load()
profile = loader.profile() 

""" 
TESTE PARA API
df = loader.load(save_folder="products")
profile = loader.profile() 


# "index = false" - diz ao pandas para não salvar a coluna de indice do Dataframe como uma coluna no arquivo CSV
df.to_csv("../data/test_dataset.csv", index=False)
"""

print(df.shape)
print(profile.keys())


print(df.head())
# esperado: n_rows, numeric_cols, categorical_cols, missing_per_col, unique_per_col, dtypes