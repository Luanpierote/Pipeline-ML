""" CARREGA OS DADOS """
# EM DESENVOLVIMENTO ⌛
from sklearn.pipeline import Pipeline
import requests
import os
import json
import pandas as pd

api_users = "https://dummyjson.com/users"
api_products = "https://dummyjson.com/products"

# Criando pipelines - Em Desenvolvimento⌛
# numeric_transformer = Pipeline(steps = [('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('ohe', OneHotCategoricalEncoder())])

def extract_data(endpoint):
  response = requests.get(endpoint)
  if response.status_code == 200:
    return response.json()
  else:
    print(f"Erro ao acessar a API. Código de status: {response.status_code}")
    return None

# Pega o Json e carrega em outra pasta
def load_data(data,path):
  # Cria a pasta caso não exista
  os.makedirs(path, exist_ok=True)
  # dentro da pasta criar um arquivo JSON com o nome do arquivo sendo o id o nome do arquivo.json
  with open(f"{path}/{data['id']}.json","w") as f:
    # Converte o json para um arquivo
    json.dump(data,f)


def load_loop_data(endpoint):
  url = "https://dummyjson.com/" + endpoint
  # ex: df_users = pd.DataFrame(extract_data(api_users)["users"])
  df = pd.DataFrame(extract_data(url)[endpoint])

  if df.empty:
    print(f"Erro ao extrair a API: {url}")
    return

  # Para iterar o dicionário e preencher o arquivo com todos os ids
  for _, row in df.iterrows():
    load_data(row.to_dict(), endpoint)

endpoints = ["users","products"]

for endpoint in endpoints:
    load_loop_data(endpoint)

# Visualização dos arquivos JSON em dataframe
def read_folder(path):
    data = []
    for file in os.listdir(path):
        with open(f"{path}/{file}", "r") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)



df = read_folder("products")
print(df.iloc[:,:])