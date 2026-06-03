# Correções realizadas no projeto

## 1. MLAgent
- O `config.json` agora é gerado com `preprocessing.encode` e `preprocessing.scale`.
- A coluna alvo não é mais incluída automaticamente no mapa de encoding.

## 2. DataPreprocessor
- Removido código solto que executava automaticamente ao importar o arquivo.
- Corrigido o uso de `drop_cols`, `impute`, `preprocessing.encode` e `preprocessing.scale`.
- Corrigido o erro de chamada `scale(method=...)`; agora usa o mapa correto de escala.
- `encode` agora aceita dicionário por coluna, como `{ "coluna": "label" }` ou `{ "coluna": "onehot" }`.
- `scale` agora aceita strings do config: `standard` e `minmax`.
- A coluna alvo é preservada no pré-processamento.

## 3. DataLoader
- Agora suporta três tipos de entrada:
  - API/URL JSON
  - pasta com arquivos `.json`
  - arquivo `.csv`

## 4. PipelineRunner
- Corrigida a execução de modelos supervisionados e não supervisionados.
- Para modelos com `target_column`, o pipeline separa `X` e `y`, treina com `fit(X, y)` e gera `predict(X)`.
- Para modelos sem `target_column`, usa `fit_predict(X)` quando disponível.
- Garante que as features enviadas ao sklearn sejam numéricas.

## 5. Validação rápida
- Os arquivos principais foram compilados com `python -m py_compile` sem erros.
- Foi feito teste manual com CSV e com modelo RandomForestClassifier.
