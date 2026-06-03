"""
Valida esquema, tipos e ranges do DataFrame contra as regras do config.
Etapa 2 do pipeline: loader → validator → preprocessor → modelo
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Armazena erros (bloqueantes) e avisos (informativos) da validação."""

    is_valid: bool = True
    errors:   list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self):
        status = "✅ VÁLIDO" if self.is_valid else "❌ INVÁLIDO"
        print(f"Status: {status}")
        for e in self.errors:   print(f"  ERRO:  {e}")
        for w in self.warnings: print(f"  AVISO: {w}")
        if not self.errors and not self.warnings:
            print("  Nenhum problema encontrado.")


class DataValidator:
    """
    Aplica as regras do config ao DataFrame e devolve um ValidationResult.

    Regras lidas do config
    ----------------------
    required  : list[str]               — colunas que devem existir
    critical  : list[str]               — colunas que não podem ter nulos
    ranges    : {col: [min, max]}       — limites numéricos aceitáveis
    unique    : list[str]               — colunas sem duplicatas permitidas
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df     = df
        self.config = config

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        df     = self.df

        if df.empty:
            result.add_error("DataFrame vazio")
            return result
        
        for col in self.config.get("required", []):
            if col not in df.columns:
                result.add_error(f"Coluna ausente: '{col}'")

        for col in self.config.get("critical", []):
            if col in df.columns and df[col].isna().any():
                result.add_error(f"Coluna '{col}' contém valores nulos")

        for col, (vmin, vmax) in self.config.get("ranges", {}).items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if vmin is not None and (s < vmin).any():
                result.add_error(f"'{col}' contém valores abaixo de {vmin}")
            if vmax is not None and (s > vmax).any():
                result.add_error(f"'{col}' contém valores acima de {vmax}")

        unique_cols = self.config.get("unique", [])
        if unique_cols and df.duplicated(subset=unique_cols).any():
            result.add_warning(f"Duplicatas encontradas nas colunas: {unique_cols}")

        return result