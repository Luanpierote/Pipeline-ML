""" VALIDA ESQUEMA,TIPOS,RANGES - Lê e reporta"""
# ETAPA CONCLUÍDA✅
import pandas as pd
import requests
from dataclasses import dataclass,field

# O processo está manual⌛
response = requests.get('https://dummyjson.com/products?limit=100')
df = pd.DataFrame(response.json()['products'])

print(f'Shape: {df.shape}')
df[['id', 'title', 'price', 'rating', 'stock', 'discountPercentage']].head()


# Armazena o resultado da validação, separando erros (bloqueantes) de avisos (informativos)
@dataclass
class ValidationResult:
# Começa como True; vira False assim que qualquer erro for registrado

  is_valid: bool = True
  # Problemas que impedem o pipeline de continuar
  errors: list = field(default_factory=list)
  # Problemas que merecem atenção, mas não bloqueiam o pipeline
  warnings: list = field(default_factory=list)

  def add_error(self, msg):
      # Registra o erro e marca o resultado como inválido
      self.errors.append(msg)
      self.is_valid = False

  def add_warning(self, msg):
      # Registra o aviso sem invalidar o resultado
      self.warnings.append(msg)

  def summary(self):
      # Exibe o diagnóstico completo da validação no notebook
      status = '✅ VÁLIDO' if self.is_valid else '❌ INVÁLIDO'
      print(f'Status: {status}')
      for e in self.errors:   print(f'  ERRO:  {e}')
      for w in self.warnings: print(f'  AVISO: {w}')
      if not self.errors and not self.warnings:
          print('  Nenhum problema encontrado.')

# Aplica as regras de validação ao DataFrame e retorna um ValidationResult
class DataValidator:
  def __init__(self, df: pd.DataFrame, config: dict):
      # Recebe o DataFrame carregado na etapa de ingestão
      self.df = df
      self.config = config

  def validate(self) -> ValidationResult:
      result = ValidationResult()
      df = self.df


      # 1. Garante que as colunas essenciais para o pipeline existem no dataset
      for col in self.config.get('required', []):
            if col not in df.columns:
                result.add_error(f"Coluna ausente: '{col}'")

      # 2. Colunas críticas não podem ter nulos — causariam falha silenciosa no modelo
      for col in self.config.get('critical', []):
            if col in df.columns and df[col].isna().any():
                result.add_error(f"Coluna '{col}' contém valores nulos")

      # 3. Verifica se os valores numéricos estão dentro dos limites esperados pelo domínio
      #    ex: preço negativo ou rating > 5 indicam erro de cadastro
      for col, (vmin, vmax) in self.config.get('ranges', {}).items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors='coerce')
            if vmin is not None and (s < vmin).any():
                result.add_error(f"'{col}' contém valores abaixo de {vmin}")
            if vmax is not None and (s > vmax).any():
                result.add_error(f"'{col}' contém valores acima de {vmax}")

      # 4. IDs duplicados indicam registros repetidos que distorceriam o treinamento
      if df.duplicated(subset=self.config.get('unique', [])).any():
            result.add_warning('Existem IDs duplicados')


      return result