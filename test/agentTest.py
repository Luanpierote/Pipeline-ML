# 6* ETAPA - FUNCIONA
# Ele tem que receber a fonte de dados, não o dataframe(O dataframe só deve existir depois do load)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from main import MLAgent

agent = MLAgent("../data/test_dataset.csv")
agent.analyze()
agent.configure()

import json
# Serve para converter um objeto python( ex.:dicionário ou lista) em uma string no formato JSON(serialização)
print(json.dumps(agent.config, indent = 2)) 
# verificar: source, required, critical, model aninhado, drop_cols, encode, scale